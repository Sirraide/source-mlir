#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/Compression.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <source/Core.hh>
#include <source/Frontend/AST.hh>

/// === INVARIANTS ===
///
/// - All strings are written as u64 names + u8[] data.
/// - isz and usz are written as u64.
///
/// Prefixes/abbreviations that are already in use.
///
/// Some of these make sense (e.g. 'C' for ‘closure’), others were just
/// chosen arbitrarily because the corresponding letter was already taken
/// (e.g. ‘X’ for constructors).
///
/// - A
/// - C
/// - E
/// - I
/// - J
/// - L
/// - M
/// - O
/// - P
/// - Q
/// - R
/// - S
/// - U
/// - W
/// - X
/// - Y
///
/// - b
/// - i
/// - n
/// - q
/// - r
/// - v

namespace src {
/// ===========================================================================
///  Mangler.
/// ===========================================================================
auto Type::_mangled_name() -> std::string {
    auto FormatSEType = [&](Expr* t, std::string_view prefix) {
        auto se = cast<SingleElementTypeBase>(t);
        return fmt::format("{}{}", prefix, se->elem.mangled_name);
    };

    auto MangleNamedType = [&](Named* s, std::string_view prefix) {
        if (s->mangled_name.empty()) {
            if (not s->module or not s->module->is_logical_module) {
                s->mangled_name = fmt::format("{}{}{}", prefix, s->name.size(), s->name);
            } else {
                s->mangled_name = fmt::format(
                    "M{}{}{}{}{}",
                    s->module->name.size(),
                    s->module->name,
                    prefix,
                    s->name.size(),
                    s->name
                );
            }
        }

        return s->mangled_name;
    };

    switch (ptr->kind) {
        case Expr::Kind::BuiltinType:
            switch (cast<BuiltinType>(ptr)->builtin_kind) {
                case BuiltinTypeKind::Void: return "v";
                case BuiltinTypeKind::Int: return "i";
                case BuiltinTypeKind::Bool: return "b";
                case BuiltinTypeKind::NoReturn: return "r";

                case BuiltinTypeKind::ArrayLiteral:
                case BuiltinTypeKind::MemberProc:
                case BuiltinTypeKind::OverloadSet:
                case BuiltinTypeKind::Unknown:
                    Unreachable("Builtin type must be resolved before mangling");
            }

            Unreachable();

        case Expr::Kind::Nil:
            return "n";

        /// The underscore is so we know how many digits are part of the bit width.
        case Expr::Kind::IntType:
            return fmt::format("I{}_", cast<IntType>(ptr)->size.bits());

        case Expr::Kind::ReferenceType: return FormatSEType(ptr, "R");
        case Expr::Kind::ScopedPointerType: return FormatSEType(ptr, "U");
        case Expr::Kind::SliceType: return FormatSEType(ptr, "L");
        case Expr::Kind::OptionalType: return FormatSEType(ptr, "O");
        case Expr::Kind::ClosureType: return FormatSEType(ptr, "C");

        case Expr::Kind::ArrayType: {
            auto a = cast<ArrayType>(ptr);
            return fmt::format("A{}{}", a->dimension(), a->elem.mangled_name);
        }

        case Expr::Kind::SugaredType:
        case Expr::Kind::ScopedType:
            return cast<SingleElementTypeBase>(ptr)->elem.mangled_name;

        case Expr::Kind::ProcType: {
            auto p = cast<ProcType>(ptr);

            /// We don’t include the return type since you can’t
            /// overload on that anyway.
            std::string name{"P"};
            if (p->variadic) name += "q";
            for (auto a : p->param_types) name += a.mangled_name;
            name += "E";
            return name;
        }

        case Expr::Kind::OpaqueType:
            return MangleNamedType(cast<OpaqueType>(ptr), "Q");

        case Expr::Kind::StructType:
            return MangleNamedType(cast<StructType>(ptr), "S");

        case Expr::Kind::TupleType: Todo("Mangle tuple type");

#define SOURCE_AST_EXPR(name) case Expr::Kind::name:
#define SOURCE_AST_TYPE(...)
#include <source/Frontend/AST.def>
            Unreachable("Not a type");
    }
}

auto ObjectDecl::_mangled_name() -> StringRef {
    auto& s = stored_mangled_name;
    if (not s.empty()) return s;

    /// Determine mangling.
    switch (mangling) {
        case Mangling::None: return name;

        /// Actually compute the mangled name.
        case Mangling::Source: break;
    }

    /// Append prefix.
    s = "_S";
    if (module->is_logical_module)
        s += fmt::format("M{}{}", module->name.size(), module->name);

    /// Procedure.
    if (auto proc = dyn_cast<ProcDecl>(this)) {
        /// Nested functions start with 'L' and the name of the parent function. Exclude
        /// the '_S' prefix from the parent function’s name.
        auto ty = cast<ProcType>(proc->type);
        if (proc->parent != proc->module->top_level_func)
            s += fmt::format("J{}", proc->parent->mangled_name.drop_front(2));

        /// Special members receive an extra sigil followed by the parent
        /// struct name and have no name themselves.
        if (proc->is_smp) {
            if (ty->smp_kind == SpecialMemberKind::Constructor)
                s += fmt::format("X{}", Type{ty->smp_parent}.mangled_name);
            else if (ty->smp_kind == SpecialMemberKind::Destructor)
                s += fmt::format("Y{}", Type{ty->smp_parent}.mangled_name);
        }

        /// Member functions have both as sigil and a name.
        else if (proc->parent_struct) {
            s += fmt::format(
                "W{}{}{}",
                Type{proc->parent_struct}.mangled_name,
                name.size(),
                name
            );
        }

        /// All other functions just include the name.
        else { s += fmt::format("{}{}", name.size(), name); }

        /// The type is always included.
        s += type.mangled_name;
        return s;
    } else {
        Todo();
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
    Opaque,

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
    llvm::DenseMap<TypeBase*, TD, TypeBase::DenseMapInfo> type_map{};

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
                SerialiseType(cast<TypeBase>(e->type));

        /// Serialise top-level function.
        SerialiseType(cast<TypeBase>(mod->top_level_func->type));
        SerialiseDecl(mod->top_level_func);

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
    auto SerialiseType(TypeBase* t) -> TD {
        const auto FindTD = [&] -> TD {
            if (auto td = type_map.find(t); td != type_map.end()) return td->second;
            return TD{};
        };

        switch (t->kind) {
            default: Unreachable();

            case Expr::Kind::SugaredType:
                return SerialiseType(cast<TypeBase>(cast<SugaredType>(t)->elem));

            case Expr::Kind::BuiltinType: {
                switch (cast<BuiltinType>(t)->builtin_kind) {
                    case BuiltinTypeKind::Void: return TD(SerialisedTypeTag::Void);
                    case BuiltinTypeKind::Int: return TD(SerialisedTypeTag::Int);
                    case BuiltinTypeKind::Bool: return TD(SerialisedTypeTag::Bool);
                    case BuiltinTypeKind::NoReturn: return TD(SerialisedTypeTag::NoReturn);

                    case BuiltinTypeKind::ArrayLiteral:
                    case BuiltinTypeKind::OverloadSet:
                    case BuiltinTypeKind::MemberProc:
                    case BuiltinTypeKind::Unknown:
                        Unreachable();
                }

                Unreachable();
            }

            case Expr::Kind::IntType: {
                auto i = cast<IntType>(t);
                switch (i->size.bits()) {
                    default: break;
                    case 8: return TD(SerialisedTypeTag::I8);
                    case 16: return TD(SerialisedTypeTag::I16);
                    case 32: return TD(SerialisedTypeTag::I32);
                    case 64: return TD(SerialisedTypeTag::I64);
                }

                if (auto td = FindTD(); td != TD{}) return td;
                *this << SerialisedTypeTag::SizedInteger;
                *this << i->size.bits();
                return type_map[t] = TD{hdr.type_count++};
            }

            case Expr::Kind::ReferenceType:
            case Expr::Kind::ScopedPointerType:
            case Expr::Kind::SliceType:
            case Expr::Kind::ArrayType:
            case Expr::Kind::OptionalType: {
                if (auto td = FindTD(); td != TD{}) return td;
                auto elem = SerialiseType(cast<TypeBase>(cast<SingleElementTypeBase>(t)->elem));
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
                if (auto a = dyn_cast<ArrayType>(t)) *this << a->dimension().getZExtValue();
                return type_map[t] = TD{hdr.type_count++};
            }

            case Expr::Kind::OpaqueType: {
                if (auto td = FindTD(); td != TD{}) return td;
                auto s = cast<OpaqueType>(t);
                *this << SerialisedTypeTag::Opaque;
                *this << ConvertMangling(s->mangling);
                *this << s->name;
                return type_map[t] = TD{hdr.type_count++};
            }

            /// TODO: Do we at all care about the static chain here?
            case Expr::Kind::ProcType: {
                if (auto td = FindTD(); td != TD{}) return td;
                auto p = cast<ProcType>(t);

                auto ret = SerialiseType(cast<TypeBase>(p->ret_type));
                std::vector<TD> params;
                for (auto a : p->param_types) params.push_back(SerialiseType(cast<TypeBase>(a)));

                *this << SerialisedTypeTag::Procedure;
                *this << ret << params.size() << p->variadic;
                for (auto td : params) *this << td;
                return type_map[t] = TD{hdr.type_count++};
            }

            case Expr::Kind::StructType: {
                if (auto td = FindTD(); td != TD{}) return td;
                auto s = cast<StructType>(t);
                if (not s->member_procs.empty() or not s->initialisers.empty()) Todo("Serialise member functions");

                /// Create struct type descriptor first to support recursive
                /// types; write name, size, alignment, and number of fields.
                type_map[t] = hdr.type_count++;
                *this << SerialisedTypeTag::Struct;
                *this << ConvertMangling(s->mangling);
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
                    *this << f->padding << f->offset;
                    if (not f->padding) *this << f->name;
                }

                /// Write field types.
                for (const auto& [i, f] : llvm::enumerate(s->all_fields)) {
                    auto td = SerialiseType(cast<TypeBase>(f->type));
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
            *this << type_map[cast<TypeBase>(p->type)];
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
    requires (std::is_trivially_copyable_v<T> and not utils::string_like<T>)
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

    auto operator<<(utils::string_like auto sv) -> Serialiser& {
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
    Module* mod{};
    UncompressedHeader uhdr;

    SmallVector<std::pair<Type*, TD>> type_fixup_list;
    SmallVector<Type> types;

    Deserialiser(Context* ctx, StringRef module_name, Location loc, ArrayRef<u8> description)
        : ctx{ctx}, loc{loc}, description{description} {
        mod = Module::Create(ctx, module_name, false, loc);
    }

    /// Abort due to ill-formed module description.
    template <typename... Args>
    [[noreturn]] void Fatal(fmt::format_string<Args...> fmt, Args&&... args) {
        std::string s = fmt::format("Module description for '{}' is ill-formed: ", mod->name);
        s += fmt::format(fmt, std::forward<Args>(args)...);
        Diag::Fatal("{}", s);
    }

    /// Entry.
    auto deserialise() -> Module* {
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
        return mod;
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
        for (usz i = 0; i < hdr.type_count; i++) types.push_back(DeserialiseType());

        /// Fixup types.
        for (auto [t, td] : type_fixup_list) *t = Map(td);

        /// Read decls.
        for (usz i = 0; i < hdr.decl_count; i++) DeserialiseDecl();
    }

    template <std::derived_from<SingleElementTypeBase> T>
    auto CreateSEType() -> TypeBase* {
        return new (&*mod) T(Map(rd<TD>()), {});
    }

    /// Map a type descriptor to an already deserialised type.
    auto Map(TD t) -> Type {
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
            }
        }

        Assert(t.index() < types.size());
        return types[t.index()];
    }

    auto DeserialiseType() -> Type {
        auto ty = DeserialiseTypeImpl();
        ty->sema.set_done();
        return ty;
    }

    auto DeserialiseTypeImpl() -> Type {
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

            case SerialisedTypeTag::SizedInteger: {
                auto bits = rd<u64>();
                switch (bits) {
                    default: break;
                    case 8: return Type::I8;
                    case 16: return Type::I16;
                    case 32: return Type::I32;
                    case 64: return Type::I64;
                }

                return new (&*mod) IntType(Size::Bits(bits), {});
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
                    new (&*mod) ConstExpr(nullptr, EvalResult{APInt(64, usz(dim)), Type::Int}, {}),
                    {}
                );
            }

            case SerialisedTypeTag::Opaque: {
                auto m = ConvertMangling(rd<SerialisedMangling>());
                auto name = rd<std::string>();
                return new (&*mod) OpaqueType(&*mod, mod->save(name), m, {});
            }

            case SerialisedTypeTag::Procedure: {
                Type ret = Map(rd<TD>());
                u64 params = rd<u64>();
                bool variadic = rd<bool>();

                SmallVector<Type> param_types{};
                for (u64 i = 0; i < params; i++) param_types.push_back(Map(rd<TD>()));
                return new (&*mod) ProcType(std::move(param_types), ret, variadic, {});
            }

            case SerialisedTypeTag::Struct: {
                auto mangling = ConvertMangling(rd<SerialisedMangling>());
                auto name = rd<std::string>();
                auto size = rd<Size>();
                auto align = rd<Align>();
                auto field_count = rd<u64>();

                /// Read field types.
                SmallVector<FieldDecl*> fields{};
                SmallVector<TD> field_types{};
                for (usz i = 0; i < field_count; i++) {
                    /// TD may refer to type that was serialised later on.
                    auto td = rd<TD>();
                    field_types.push_back(td);
                    fields.push_back(new (&*mod) FieldDecl(
                        String(),
                        td.is_builtin() or td.index() < types.size()
                            ? Map(field_types.back())
                            : Type::UnsafeEmpty(),
                        {},
                        {},
                        u32(i)
                    ));
                }

                /// Read fields.
                for (auto& f : fields) {
                    f->padding = rd<bool>();
                    f->offset = rd<Size>();
                    if (not f->padding) f->name = mod->save(rd<std::string>());
                }

                /// Create the struct.
                auto s = new (&*mod) StructType(
                    &*mod,
                    mod->save(name),
                    std::move(fields),
                    {},
                    {},
                    {},
                    new (&*mod) BlockExpr(&*mod, mod->global_scope),
                    mangling,
                    {}
                );

                /// Mark fields for fixup if need be.
                for (auto& f : s->all_fields)
                    if (static_cast<Expr*>(f->type) == nullptr)
                        type_fixup_list.emplace_back(&f->stored_type, field_types[f->index]);

                /// Set size and alignment.
                s->stored_alignment = align;
                s->stored_size = size;
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
                mod->exports[cast<StructType>(ty)->name].push_back(static_cast<Expr*>(ty));
                return;
            }

            case SerialisedDeclTag::Procedure: {
                auto m = ConvertMangling(rd<SerialisedMangling>());
                auto name = rd<std::string>();
                auto ty = Map(rd<TD>());
                auto p = new (&*mod) ProcDecl(
                    &*mod,
                    mod->top_level_func,
                    mod->save(name),
                    ty,
                    {},
                    Linkage::Imported,
                    m,
                    {}
                );

                p->sema.set_done();
                mod->exports[p->name].push_back(p);
                return;
            }
        }

        Unreachable("Invalid SerialisedDeclTag: {}", +tag);
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
    requires (std::is_trivially_copyable_v<T> and not utils::string_like<T>)
    auto operator>>(T& t) -> Deserialiser& {
        if (description.size() < sizeof(T)) {
            Diag::Fatal(
                "Ill-formed module description for '{}': expected {} bytes, got {}",
                mod->name,
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
                "Ill-formed module description for '{}': expected {} bytes, got {}",
                mod->name,
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
        str.resize_and_overwrite(sz, [&](char* data, usz allocated) {
            Assert(allocated >= sz, "std::string::resize_and_overwrite() is broken");
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
    StringRef module_name,
    Location loc,
    ArrayRef<u8> description
) -> Module* {
    return Deserialiser{ctx, module_name, loc, description}.deserialise();
}
