#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/Compression.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <source/Core.hh>
#include <source/Frontend/AST.hh>

/// === INVARIANTS ===
///
/// - All strings are written as u64 names + u8[] data.
/// - isz and usz are written as u64.

namespace src {
/// ===========================================================================
///  Mangler.
/// ===========================================================================
auto Expr::TypeHandle::mangled_name(Context* ctx) -> std::string {
    auto FormatSEType = [&](Expr* t, std::string_view prefix) {
        auto se = cast<SingleElementTypeBase>(t);
        return fmt::format("{}{}", prefix, se->elem->as_type.mangled_name(ctx));
    };

    switch (ptr->kind) {
        case Expr::Kind::BuiltinType:
            switch (cast<BuiltinType>(ptr)->builtin_kind) {
                case BuiltinTypeKind::Unknown: return "<invalid>";
                case BuiltinTypeKind::Void: return "v";
                case BuiltinTypeKind::Int: return "i";
                case BuiltinTypeKind::Bool: return "b";
                case BuiltinTypeKind::NoReturn: return "n";
            }

            Unreachable();

        /// Need to keep these since we allow overloading on them.
        case Expr::Kind::FFIType: {
            switch (cast<FFIType>(ptr)->ffi_kind) {
                case FFITypeKind::CChar: return "Fc";
                case FFITypeKind::CInt: return "Fi";
            }

            Unreachable();
        }

        case Expr::Kind::IntType: return fmt::format("I{}", cast<IntType>(ptr)->bits);

        case Expr::Kind::ReferenceType: return FormatSEType(ptr, "R");
        case Expr::Kind::ScopedPointerType: return FormatSEType(ptr, "U");
        case Expr::Kind::SliceType: return FormatSEType(ptr, "S");
        case Expr::Kind::ArrayType: return FormatSEType(ptr, "A");
        case Expr::Kind::OptionalType: return FormatSEType(ptr, "O");

        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            return cast<SingleElementTypeBase>(ptr)->elem->as_type.mangled_name(ctx);

        case Expr::Kind::ProcType: {
            auto p = cast<ProcType>(ptr);
            Assert(not p->static_chain_parent, "Cannot mangle local function type");

            /// We don’t include the return type since you can’t
            /// overload on that anyway.
            std::string name{"P"};
            name += fmt::format("{}", p->param_types.size());
            for (auto a : p->param_types) name += a->as_type.mangled_name(ctx);
            return name;
        }

        /// Context may be null in this case only.
        case Expr::Kind::StructType: {
            auto s = cast<StructType>(ptr);
            if (s->mangled_name.empty()) {
                if (not s->module or not s->module->is_logical_module) {
                    s->mangled_name = fmt::format("{}{}", s->name.size(), s->name);
                } else {
                    s->mangled_name = fmt::format(
                        "M{}{}{}{}",
                        s->module->name.size(),
                        s->module->name,
                        s->name.size(),
                        s->name
                    );
                }
            }

            return s->mangled_name;
        }

        case Expr::Kind::AssertExpr:
        case Expr::Kind::ReturnExpr:
        case Expr::Kind::DeferExpr:
        case Expr::Kind::WhileExpr:
        case Expr::Kind::ExportExpr:
        case Expr::Kind::LoopControlExpr:
        case Expr::Kind::BlockExpr:
        case Expr::Kind::InvokeExpr:
        case Expr::Kind::ConstExpr:
        case Expr::Kind::CastExpr:
        case Expr::Kind::MemberAccessExpr:
        case Expr::Kind::ScopeAccessExpr:
        case Expr::Kind::UnaryPrefixExpr:
        case Expr::Kind::IfExpr:
        case Expr::Kind::BinaryExpr:
        case Expr::Kind::DeclRefExpr:
        case Expr::Kind::ModuleRefExpr:
        case Expr::Kind::LocalRefExpr:
        case Expr::Kind::BoolLiteralExpr:
        case Expr::Kind::IntegerLiteralExpr:
        case Expr::Kind::StringLiteralExpr:
        case Expr::Kind::LocalDecl:
        case Expr::Kind::ProcDecl:
            Unreachable("Not a type");
    }
}

/// ===========================================================================
///  Serialisation Format.
/// ===========================================================================
namespace {
using magic_t = std::array<u8, 3>;

struct CompressedHeaderV0 {
    u64 type_count; ///< Number of types in the module.
    u64 decl_count; ///< Number of declarations in the module.
};

/// These are part of the format and must *not* be changed. Adding
/// a new enum member here requires incrementing the format version
/// number.
enum struct SerialisedDeclTag : u8 {
    StructType,
    Procedure,
};

enum struct SerialisedMangling : u8 {
    None,
    Source,
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
concept serialisable_enum = is_same<
    T,
    SerialisedMangling,
    SerialisedDeclTag,
    SerialisedTypeTag>;

constexpr u8 current_version = 0;
constexpr magic_t src_magic{'S', 'R', 'C'};

/// Module description header.
struct UncompressedHeader {
    u8 version;    ///< Version number for backwards-compatibility.
    magic_t magic; ///< Magic number.
    u64 size;      ///< Size of the uncompressed module description, excluding the header.

    UncompressedHeader() = default;
    UncompressedHeader(usz uncompressed_description_size)
        : version{current_version}, magic{src_magic}, size{uncompressed_description_size} {}
};

/// ===========================================================================
///  Serialiser.
/// ===========================================================================
struct CompareTypes {
    static bool operator()(Type* a, Type* b) { return Type::Equal(a, b); }
};

/// Type descriptor.
///
/// Builtin types are allocated between 0 and 255, so we
/// have to subtract 255 from the type index to get the
/// actual index in the type descriptor table.
class TD {
    static constexpr u64 StartVal = 256;
    static constexpr u64 InvalidVal = ~0zu;

public:
    u64 raw;

    constexpr TD() : raw{InvalidVal} {}
    constexpr TD(u64 idx) : raw{idx + StartVal} {}
    constexpr TD(SerialisedTypeTag tag) : raw{static_cast<u64>(tag)} {}

    constexpr auto operator<=>(const TD&) const = default;
    constexpr bool operator==(const TD&) const = default;

    /// Check if this is a builtin type.
    [[nodiscard]] constexpr bool is_builtin() const { return raw < StartVal; }

    /// Get the builtin type tag.
    [[nodiscard]] constexpr auto builtin_tag() const -> SerialisedTypeTag {
        Assert(is_builtin());
        return static_cast<SerialisedTypeTag>(raw);
    }

    /// Get the type index.
    [[nodiscard]] constexpr auto index() const -> u64 {
        Assert(not is_builtin());
        return raw - StartVal;
    }
};

struct Serialiser {
    Module* const mod;
    std::vector<u8> out{};

    /// Map from types to indices.
    std::unordered_map<Type*, TD, std::hash<Type*>, CompareTypes> type_map{};

    /// Larger section of the header.
    CompressedHeaderV0 hdr{
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
                SerialiseDecl(e);

        /// Update compressed header.
        std::memcpy(out.data(), &hdr, sizeof(CompressedHeaderV0));

        /// Write uncompressed header.
        UncompressedHeader uhdr{out.size()};
        SmallVector<u8> compressed{};
        compressed.resize_for_overwrite(sizeof(uhdr));
        std::memcpy(compressed.data(), &uhdr, sizeof(uhdr));

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
            type_map[t] = hdr.type_count++;
            return TD{};
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

                if (auto td = AllocateTD(); td != TD{}) return td;
                *this << SerialisedTypeTag::SizedInteger;
                *this << i->bits;
                return type_map[t];
            }

            case Expr::Kind::ReferenceType:
            case Expr::Kind::ScopedPointerType:
            case Expr::Kind::SliceType:
            case Expr::Kind::ArrayType:
            case Expr::Kind::OptionalType: {
                if (auto td = AllocateTD(); td != TD{}) return td;
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

                /// Also write the size, if applicable.
                if (auto a = dyn_cast<ArrayType>(t)) *this << u64(a->dimension());

                return type_map[t];
            }

            /// TODO: Do we at all care about the static chain here?
            case Expr::Kind::ProcType: {
                if (auto td = AllocateTD(); td != TD{}) return td;
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
                if (auto td = AllocateTD(); td != TD{}) return td;
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
                for (auto& f : s->all_fields) {
                    *this << f.padding << f.offset;
                    if (not f.padding) *this << f.name;
                }

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
    void SerialiseDecl(Expr* e) {
        /// Exported types have already been serialised; just
        /// point to the type descriptor here.
        if (auto t = dyn_cast<StructType>(e)) {
            hdr.decl_count++;
            *this << SerialisedDeclTag::StructType;
            *this << type_map[t];
            return;
        }

        if (auto p = dyn_cast<ProcDecl>(e)) {
            hdr.decl_count++;
            *this << SerialisedDeclTag::Procedure;
            *this << ConvertMangling(p->mangling);
            *this << p->name;
            *this << type_map[cast<Type>(p->type)];
            return;
        }

        Unreachable();
    }

    /// Convert mangling scheme to serialised mangling scheme.
    auto ConvertMangling(Mangling m) -> SerialisedMangling {
        switch (m) {
            case Mangling::None: return SerialisedMangling::None;
            case Mangling::Source: return SerialisedMangling::Source;
        }

        Unreachable();
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
    requires std::is_trivially_copyable_v<T>
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

/// ===========================================================================
///  Deserialiser.
/// ===========================================================================
struct Deserialiser {
    Context* ctx;
    Location loc;
    ArrayRef<u8> description;
    std::unique_ptr<Module> mod{};
    UncompressedHeader uhdr;

    SmallVector<std::pair<Expr**, TD>> type_fixup_list;
    SmallVector<Type*> types;

    Deserialiser(Context* ctx, std::string module_name, Location loc, ArrayRef<u8> description)
        : ctx{ctx}, loc{loc}, description{description} {
        mod = std::make_unique<Module>(ctx, std::move(module_name), loc);
    }

    /// Abort due to ill-formed module description.
    template <typename... Args>
    [[noreturn]] void Fatal(fmt::format_string<Args...> fmt, Args&&... args) {
        std::string s = fmt::format("Module description for '{}' is ill-formed: ", mod->name);
        s += fmt::format(fmt, std::forward<Args>(args)...);
        Diag::Fatal("{}", s);
    }

    /// Entry.
    auto deserialise() -> std::unique_ptr<Module> {
        auto data = StringRef{reinterpret_cast<const char*>(description.data()), description.size()};
        auto obj_contents = llvm::MemoryBufferRef{data, mod->name};
        auto expected_obj = llvm::object::ObjectFile::createObjectFile(obj_contents);
        if (auto err = expected_obj.takeError())
            Fatal("not a valid object file: {}", llvm::toString(std::move(err)));

        /// Get the section containing the module description.
        auto name = mod->description_section_name();
        auto obj = expected_obj->get();
        auto sect = std::find_if(obj->section_begin(), obj->section_end(), [&](const llvm::object::SectionRef& s) {
            auto sect_name = s.getName();
            if (sect_name.takeError()) return false;
            return *sect_name == name;
        });

        /// Make sure it exists.
        if (sect == obj->section_end()) Fatal("description section not found");

        /// Get the section contents.
        auto sect_contents = sect->getContents();
        if (auto err = sect_contents.takeError()) Fatal(
            "failed to get contents of description section: {}",
            llvm::toString(std::move(err))
        );

        /// Read from the section from now on.
        description = {reinterpret_cast<const u8*>(sect_contents->data()), sect_contents->size()};

        /// Read uncompressed header.
        *this >> uhdr;
        if (uhdr.magic != src_magic) Fatal("invalid magic number");

        /// Decode versioned description.
        switch (uhdr.version) {
            default: Fatal(
                "unsupported version '{}'; the latest supported version is '{}'",
                uhdr.version,
                current_version
            );

            case 0: DeserialiseV0(); break;
        }

        /// If we get here, everything went well.
        return std::move(mod);
    }

    /// Version 0.
    void DeserialiseV0() {
        /// Uncompress the header.
        SmallVector<u8> data{};
        utils::Decompress(data, description, uhdr.size);
        description = data;

        /// Extract header and name.
        auto hdr = rd<CompressedHeaderV0>();
        auto name = rd<std::string>();
        if (name != mod->name) Fatal("module name mismatch");

        /// Read types.
        ///
        /// The types are stored in a list since some types contain
        /// forward references to types defined later on, so we have
        /// to be able to map type descriptors (= indices, roughly) to
        /// types later on.
        for (usz i = 0; i < hdr.type_count; i++) {
            types.push_back(DeserialiseType());
            types.back()->sema.set_done();
        }

        /// Fixup types.
        for (auto [t, td] : type_fixup_list) *t = Map(td);

        /// Read decls.
        for (usz i = 0; i < hdr.decl_count; i++) DeserialiseDecl();
    }

    template <std::derived_from<SingleElementTypeBase> T>
    auto CreateSEType() -> Type* {
        return new (&*mod) T(Map(rd<TD>()), {});
    }

    /// Map a type descriptor to an already deserialised type.
    auto Map(TD t) -> Type* {
        if (t.is_builtin()) {
            switch (t.builtin_tag()) {
                default: Unreachable();
                case SerialisedTypeTag::Void: return Type::Void;
                case SerialisedTypeTag::Int: return Type::Int;
                case SerialisedTypeTag::Bool: return Type::Bool;
                case SerialisedTypeTag::NoReturn: return Type::NoReturn;
                case SerialisedTypeTag::I8: return Type::I8;
                case SerialisedTypeTag::I16: return Type::I16;
                case SerialisedTypeTag::I32: return Type::I32;
                case SerialisedTypeTag::I64: return Type::I64;
                case SerialisedTypeTag::CChar: return Type::CChar;
                case SerialisedTypeTag::CInt: return Type::CInt;
            }
        }

        Assert(t.index() < types.size());
        return types[t.index()];
    }

    auto DeserialiseType() -> Type* {
        auto tag = rd<SerialisedTypeTag>();
        switch (tag) {
            /// Unfortunately, we have to deserialise these and
            /// store them in the type list if they’ve actually
            /// been written to the description since the indices
            /// of types later on need to be correct.
            case SerialisedTypeTag::Void: return Type::Void;
            case SerialisedTypeTag::Int: return Type::Int;
            case SerialisedTypeTag::Bool: return Type::Bool;
            case SerialisedTypeTag::NoReturn: return Type::NoReturn;
            case SerialisedTypeTag::I8: return Type::I8;
            case SerialisedTypeTag::I16: return Type::I16;
            case SerialisedTypeTag::I32: return Type::I32;
            case SerialisedTypeTag::I64: return Type::I64;
            case SerialisedTypeTag::CChar: return Type::CChar;
            case SerialisedTypeTag::CInt: return Type::CInt;

            case SerialisedTypeTag::SizedInteger: {
                auto bits = rd<u64>();
                switch (bits) {
                    default: break;
                    case 8: return Type::I8;
                    case 16: return Type::I16;
                    case 32: return Type::I32;
                    case 64: return Type::I64;
                }

                return new (&*mod) IntType(isz(bits), {});
            }

            case SerialisedTypeTag::Reference: return CreateSEType<ReferenceType>();
            case SerialisedTypeTag::ScopedPointer: return CreateSEType<ScopedPointerType>();
            case SerialisedTypeTag::Slice: return CreateSEType<SliceType>();
            case SerialisedTypeTag::Optional: return CreateSEType<OptionalType>();

            case SerialisedTypeTag::Array: {
                auto elem = Map(rd<TD>());
                auto dim = isz(rd<u64>());
                return new (&*mod) ArrayType(
                    elem,
                    new (&*mod) ConstExpr(new (&*mod) IntLitExpr(dim, {}), EvalResult{dim}, {}),
                    {}
                );
            }

            case SerialisedTypeTag::Procedure: {
                Type* ret = Map(rd<TD>());
                u64 params = rd<u64>();

                SmallVector<Expr*> param_types{};
                param_types.resize(params);
                for (auto& t : param_types) t = Map(rd<TD>());

                return new (&*mod) ProcType(std::move(param_types), ret, {});
            }

            case SerialisedTypeTag::Struct: {
                auto name = rd<std::string>();
                auto size = rd<u64>();
                auto align = rd<u64>();
                auto field_count = rd<u64>();

                /// Read field types.
                SmallVector<StructType::Field> fields{};
                SmallVector<Type*> field_types{};
                fields.resize(field_count);
                field_types.resize(field_count);
                for (usz i = 0; i < field_count; i++) {
                    /// TD may refer to type that was serialised later on.
                    if (auto td = rd<TD>(); td.is_builtin() or td.index() < types.size()) field_types[i] = Map(td);
                    else type_fixup_list.push_back({&fields[i].type, td});
                    fields[i].index = u32(i);
                }

                /// Read fields.
                for (auto& f : fields) {
                    f.padding = rd<bool>();
                    f.offset = isz(rd<u64>());
                    if (not f.padding) f.name = rd<std::string>();
                    f.type = field_types[f.index];
                }

                /// Create the struct.
                auto s = new (&*mod) StructType(
                    &*mod,
                    std::move(name),
                    std::move(fields),
                    new (&*mod) Scope(mod->global_scope, &*mod),
                    {}
                );

                /// Set size and alignment.
                s->stored_alignment = isz(align);
                s->stored_size = isz(size);
                return s;
            }
        }

        Unreachable();
    }

    void DeserialiseDecl() {
        auto tag = rd<SerialisedDeclTag>();
        switch (tag) {
            /// This has already been deserialised as a type,
            /// so just get the TD, and that’s it.
            case SerialisedDeclTag::StructType: {
                auto ty = Map(rd<TD>());
                mod->exports[cast<StructType>(ty)->name].push_back(ty);
                return;
            }

            case SerialisedDeclTag::Procedure: {
                auto m = ConvertMangling(rd<SerialisedMangling>());
                auto name = rd<std::string>();
                auto ty = Map(rd<TD>());
                auto p = new (&*mod) ProcDecl(
                    &*mod,
                    mod->top_level_func,
                    std::move(name),
                    ty,
                    {},
                    Linkage::Imported,
                    m,
                    {}
                );

                mod->exports[p->name].push_back(p);
                return;
            }
        }

        Unreachable();
    }

    /// Convert serialised mangling scheme to mangling scheme.
    auto ConvertMangling(SerialisedMangling m) -> Mangling {
        switch (m) {
            case SerialisedMangling::None: return Mangling::None;
            case SerialisedMangling::Source: return Mangling::Source;
        }

        Unreachable();
    }

    /// Equivalent to `T t; *this >> t;`
    template <typename T>
    auto rd() -> T {
        T t;
        *this >> t;
        return t;
    };


    /// Disallow reading pointers and enums.
    ///
    /// See the Serialiser API for more information on this.
    template <typename T>
    requires (std::is_pointer_v<T> or (std::is_enum_v<T> and not serialisable_enum<T>))
    auto operator>>(T&) -> Serialiser& = delete;

    /// Read a type.
    template <typename T>
    requires std::is_trivially_copyable_v<T>
    auto operator>>(T& t) -> Deserialiser& {
        if (description.size() < sizeof(T)) {
            Diag::Fatal(
                "expected {} bytes, got {}",
                sizeof(T),
                description.size()
            );
        }

        std::memcpy(std::addressof(t), description.data(), sizeof(T));
        description = description.drop_front(sizeof(T));
        return *this;
    }

    /// Read a span of characters.
    template <typename T>
    requires (std::is_trivially_copyable_v<T> and not std::is_pointer_v<T>)
    auto operator>>(std::span<T> data) -> Deserialiser& {
        if (description.size() < data.size_bytes()) {
            Diag::Fatal(
                "expected {} bytes, got {}",
                data.size_bytes(),
                description.size()
            );
        }

        std::memcpy(data.data(), description.data(), data.size_bytes());
        description = description.drop_front(data.size_bytes());
        return *this;
    }

    /// Read a string.
    auto operator>>(std::string& str) -> Deserialiser& {
        u64 sz;
        *this >> sz;
        str.resize_and_overwrite(sz, [this](char* data, usz sz) {
            *this >> std::span<u8>(reinterpret_cast<u8*>(data), sz);
            return sz;
        });
        return *this;
    }
};

} // namespace
} // namespace src

auto src::Module::serialise() -> SmallVector<u8> {
    return Serialiser{this}.serialise();
}

auto src::Module::Deserialise(
    Context* ctx,
    std::string module_name,
    Location loc,
    ArrayRef<u8> description
) -> Module* {
    auto m = Deserialiser{ctx, std::move(module_name), loc, description}.deserialise();
    ctx->modules.push_back(std::move(m));
    return ctx->modules.back().get();
}