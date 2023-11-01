#include <llvm/Support/Compression.h>
#include <source/Core.hh>
#include <source/Frontend/AST.hh>
#include <zstd.h>

namespace src {
namespace {
using magic_t = std::array<u8, 3>;

struct CompressedHeader {
    u64 name_len;   ///< Module name length.
    u64 type_count; ///< Number of types in the module.
    u64 decl_count; ///< Number of declarations in the module.
};

/// These are part of the format and must *not* be changed. Adding
/// a new enum member here requires incrementing the format version
/// number.
enum struct SerialisedDeclTag : u8 {
    StructType,
};

enum struct SerialisedTypeTag : u8 {
    Struct,
    SizedInteger,
    Reference,
    ScopedPointer,
    Slice,
    Array,
    Optional,
    Procedure,

    /// The ones below here also double as type descriptors
    /// for builtin types, as those are never emitted into
    /// the type descriptor table.
    Void = 64,
    Int,
    Bool,
    NoReturn,

    /// Common int types.
    I8,
    I16,
    I32,
    I64,

    CChar = 128,
    CInt,
};

template <typename T>
concept serialisable_enum = is_same<T, SerialisedDeclTag, SerialisedTypeTag>;

static constexpr u8 current_version = 0;
static constexpr magic_t src_magic{'S', 'R', 'C'};


/// Module description header.
struct {
    u8 version;    ///< Version number for backwards-compatibility.
    magic_t magic; ///< Magic number.
} UncomprHdr{current_version, src_magic};


/// ===========================================================================
///  Serialiser.
/// ===========================================================================
struct CompareTypes {
    static bool operator()(Type* a, Type* b) { return Type::Equal(a, b); }
};

struct Serialiser {
    Module* const mod;
    std::vector<u8> out{};

    /// Type descriptor.
    ///
    /// Builtin types are allocated between 0 and 255, so we
    /// have to subtract 255 from the type index to get the
    /// actual index in the type descriptor table.
    using TD = u64;
    static constexpr TD TDStart = 256, TDInvalid = ~TD(0);

    /// Map from types to indices.
    std::unordered_map<Type*, TD, std::hash<Type*>, CompareTypes> type_map{};

    /// Larger section of the header.
    CompressedHeader hdr{
        .name_len = u32(mod->name.size()),
        .type_count = 0,
        .decl_count = 0,
    };

    /// Entry.
    auto serialise() -> SmallVector<u8> {
        *this << hdr;
        *this << mod->name;

        /// Serialise all exports’ types.
        for (auto& [_, vec] : mod->exports)
            for (auto e : vec)
                SerialiseType(cast<Type>(e->type));

        /// Serialise the exports themselves.
        for (auto& [_, vec] : mod->exports)
            for (auto e : vec)
                Serialise(e);

        /// Update compressed header.
        std::memcpy(out.data(), &hdr, sizeof(CompressedHeader));

        /// Write uncompressed header.
        SmallVector<u8> compressed{};
        compressed.resize_for_overwrite(sizeof(UncomprHdr));
        std::memcpy(compressed.data(), &UncomprHdr, sizeof(UncomprHdr));

        /// Compress the types, exports, compressed header, and name.
        utils::Compress(compressed, out);

        /// Done!
        return compressed;
    }

    /// Serialise a type.
    auto SerialiseType(Type* t) -> TD {
        const auto AllocateTD = [&] {
            /// Don’t write the same type twice.
            if (auto td = type_map.find(t); td != type_map.end()) return td->second;

            /// Type still has to be serialised.
            type_map[t] = TDStart + hdr.type_count++;
            return TDInvalid;
        };

        switch (t->kind) {
            default: Unreachable();

            case Expr::Kind::SugaredType:
                return SerialiseType(cast<Type>(cast<SugaredType>(t)->elem));

            case Expr::Kind::BuiltinType: {
                switch (cast<BuiltinType>(t)->builtin_kind) {
                    case BuiltinTypeKind::Unknown: Unreachable();
                    case BuiltinTypeKind::Void: return TD(SerialisedTypeTag::Void);
                    case BuiltinTypeKind::Int: return TD(SerialisedTypeTag::Int);
                    case BuiltinTypeKind::Bool: return TD(SerialisedTypeTag::Bool);
                    case BuiltinTypeKind::NoReturn: return TD(SerialisedTypeTag::NoReturn);
                }

                Unreachable();
            }

            case Expr::Kind::FFIType: {
                switch (cast<FFIType>(t)->ffi_kind) {
                    case FFITypeKind::CChar: return TD(SerialisedTypeTag::CChar);
                    case FFITypeKind::CInt: return TD(SerialisedTypeTag::CInt);
                }

                Unreachable();
            }

            case Expr::Kind::IntType: {
                auto i = cast<IntType>(t);
                switch (i->bits) {
                    default: break;
                    case 8: return TD(SerialisedTypeTag::I8);
                    case 16: return TD(SerialisedTypeTag::I16);
                    case 32: return TD(SerialisedTypeTag::I32);
                    case 64: return TD(SerialisedTypeTag::I64);
                }

                if (auto td = AllocateTD(); td != TDInvalid) return td;
                *this << SerialisedTypeTag::SizedInteger;
                *this << i->bits;
                return type_map[t];
            }

            case Expr::Kind::ReferenceType:
            case Expr::Kind::ScopedPointerType:
            case Expr::Kind::SliceType:
            case Expr::Kind::ArrayType:
            case Expr::Kind::OptionalType: {
                if (auto td = AllocateTD(); td != TDInvalid) return td;
                auto elem = SerialiseType(cast<Type>(cast<SingleElementTypeBase>(t)->elem));
                switch (t->kind) {
                    default: Unreachable();
                    case Expr::Kind::ReferenceType: *this << SerialisedTypeTag::Reference; break;
                    case Expr::Kind::ScopedPointerType: *this << SerialisedTypeTag::ScopedPointer; break;
                    case Expr::Kind::SliceType: *this << SerialisedTypeTag::Slice; break;
                    case Expr::Kind::ArrayType: *this << SerialisedTypeTag::Array; break;
                    case Expr::Kind::OptionalType: *this << SerialisedTypeTag::Optional; break;
                }
                *this << elem;
                return type_map[t];
            }

            /// TODO: Do we at all care about the static chain here?
            case Expr::Kind::ProcType: {
                if (auto td = AllocateTD(); td != TDInvalid) return td;
                auto p = cast<ProcType>(t);

                auto ret = SerialiseType(cast<Type>(p->ret_type));
                SmallVector<TD, 8> params;
                for (auto a : p->param_types) params.push_back(SerialiseType(cast<Type>(a)));

                *this << SerialisedTypeTag::Procedure;
                *this << ret << params.size();
                for (auto td : params) *this << td;
                return type_map[t];
            }

            case Expr::Kind::StructType: {
                if (auto td = AllocateTD(); td != TDInvalid) return td;
                auto s = cast<StructType>(t);

                /// Write name, size, alignment, and number of fields.
                *this << SerialisedTypeTag::Struct;
                *this << s->name << s->stored_size << s->stored_alignment << s->all_fields.size();

                /// Allocate space for field types. We may have to serialise
                /// these as well, but we can’t start serialising types inside
                /// of each other, so we first write the rest of the struct data,
                /// then serialise the field types after that, and then come back
                /// and fill in the proper descriptors here.
                const auto offs = out.size();
                out.resize(offs + s->all_fields.size() * sizeof(TD));

                /// Write field data, excluding the type.
                for (auto& f : s->all_fields)
                    *this << f.padding << f.offset << (f.padding ? StringRef{} : f.name);

                /// Write field types.
                for (const auto& [i, f] : vws::enumerate(s->all_fields)) {
                    auto td = SerialiseType(cast<Type>(f.type));
                    WriteAt(offs + usz(i) * sizeof(TD), td);
                }

                return type_map[t];
            }
        }
    }

    /// Serialise an export.
    void Serialise(Expr* e) {
        /// Exported types have already been serialised; just
        /// point to the type descriptor here.
        if (auto t = dyn_cast<StructType>(e)) {
            hdr.decl_count++;
            *this << SerialisedDeclTag::StructType;
            *this << type_map[t];
            return;
        }
    }

    /// Insert data at a certain offset.
    template <typename T>
    requires (std::is_trivially_copyable_v<T> and not std::is_pointer_v<T>)
    void WriteAt(usz offs, const T& data) {
        Assert(offs + sizeof(T) <= out.size());
        auto ptr = reinterpret_cast<const u8*>(std::addressof(data));
        std::copy(ptr, ptr + sizeof(T), out.begin() + isz(offs));
    }

    /// Disallow writing pointers and enums.
    ///
    /// The former need to be serialised properly, and the latter
    /// are fairly unstable and may change randomly, and we don’t
    /// want the binary format to depend on them.
    ///
    /// The exception to this is `SerialisedDeclTag`, since it is
    /// explicitly part of the binary format.
    template <typename T>
    requires (std::is_pointer_v<T> or (std::is_enum_v<T> and not serialisable_enum<T>))
    auto operator<<(T) -> Serialiser& = delete;

    /// Write bytes to a vector.
    template <typename T>
    requires (std::is_trivially_copyable_v<T>)
    auto operator<<(const T& t) -> Serialiser& {
        using Ty = std::remove_cvref_t<T>;

        /// Fix usz and isz to u64.
        if constexpr (std::is_same_v<Ty, usz> or std::is_same_v<Ty, isz>) {
            out.resize(out.size() + sizeof(u64));
            std::memcpy(out.data() + out.size() - sizeof(u64), std::addressof(t), sizeof(u64));
            return *this;
        }

        /// Copy bytes.
        auto data = reinterpret_cast<const u8*>(std::addressof(t));
        out.insert(out.end(), data, data + sizeof(T));
        return *this;
    }

    auto operator<<(StringRef sv) -> Serialiser& {
        *this << u64(sv.size());
        out.insert(out.end(), sv.begin(), sv.end());
        return *this;
    }
};
} // namespace
} // namespace src

auto src::Module::serialise() -> SmallVector<u8> {
    return Serialiser{this}.serialise();
}
