#ifndef SOURCE_MLIR_UTILS_HH
#define SOURCE_MLIR_UTILS_HH

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <coroutine>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <fmt/color.h>
#include <fmt/format.h>
#include <functional>
#include <llvm/ADT/Any.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TinyPtrVector.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/StringSaver.h>
#include <memory>
#include <mlir/Support/LogicalResult.h>
#include <new>
#include <numeric>
#include <optional>
#include <ranges>
#include <source_location>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace src {
using namespace std::literals;

namespace fs = std::filesystem;
namespace chr = std::chrono;
namespace rgs = std::ranges;
namespace vws = std::ranges::views;

using llvm::Align;
using llvm::APInt;
using llvm::ArrayRef;
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::MutableArrayRef;
using llvm::SmallString;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringMap;
using llvm::StringRef;
using llvm::Twine;
using llvm::TypeSize;

using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_if_present;
using llvm::isa;

using mlir::LogicalResult;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using usz = size_t;
using uptr = uintptr_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using isz = ptrdiff_t;
using iptr = intptr_t;

using f32 = float;
using f64 = double;

#define STR_(X) #X
#define STR(X)  STR_(X)

#define CAT_(X, Y) X##Y
#define CAT(X, Y)  CAT_(X, Y)

#define FWD(x) std::forward<decltype(x)>(x)

/// \brief Defer execution of a lambda until the end of the scope.
///
/// Example:
/// \code{.cpp}
///     auto file = std::fopen(...);
///     defer { if (file) std::fclose(file); };
/// \endcode
#define defer auto CAT(_defer_, __COUNTER__) = ::src::detail::DeferStage1{}->*[&]

/// \brief Temporarily set a variable to a value.
///
/// Example:
/// \code{.cpp}
///     static int x = 0;
///     tempset x = 1;
///     /// x is reset to `0` at end of scope.
/// \endcode
#define tempset auto CAT(_tempset_, __COUNTER__) = ::src::detail::TempsetStage1{}->*

#define readonly(type, name, code) \
    type _##name() { code; }       \
    __declspec(property(get = _##name)) type name

#define readonly_const(type, name, code) \
    type _##name() const { code; }       \
    __declspec(property(get = _##name)) type name

#define readonly_decl(type, name) \
    type _##name();               \
    __declspec(property(get = _##name)) type name

#define readonly_const_decl(type, name) \
    type _##name() const;               \
    __declspec(property(get = _##name)) type name

#define property_decl(type, name) \
    type _##name();               \
    void _set_##name(type);       \
    __declspec(property(get = _##name, put = _set_##name)) type name

// clang-format off
#define AssertImpl(kind, cond, ...) (cond ? void(0) : \
    ::src::detail::AssertFail(                        \
        ::src::detail::AssertKind::kind,              \
        #cond,                                        \
        __FILE__,                                     \
        __LINE__                                      \
        __VA_OPT__(, fmt::format(__VA_ARGS__))        \
    )                                                 \
)

#define AbortImpl(kind, ...)                    \
    ::src::detail::AssertFail(                  \
        ::src::detail::AssertKind::kind,        \
        "",                                     \
        __FILE__,                               \
        __LINE__                                \
        __VA_OPT__(, fmt::format(__VA_ARGS__))  \
    )                                           \

#define Assert(cond, ...) AssertImpl(AK_Assert, cond __VA_OPT__(, __VA_ARGS__))
#define Todo(...) AbortImpl(AK_Todo __VA_OPT__(, __VA_ARGS__))
#define Unreachable(...) AbortImpl(AK_Unreachable __VA_OPT__(, __VA_ARGS__))
// clang-format on

namespace detail {
enum struct AssertKind {
    AK_Assert,
    AK_Todo,
    AK_Unreachable,
};

[[noreturn]] void AssertFail(
    AssertKind k,
    std::string_view condition,
    std::string_view file,
    int line,
    std::string&& message = ""
);

template <typename Callable>
struct DeferStage2 {
    Callable cb;
    ~DeferStage2() { cb(); }

    explicit DeferStage2(Callable&& _cb)
        : cb(std::forward<Callable>(_cb)) {}
};

struct DeferStage1 {
    template <typename Callable>
    DeferStage2<Callable> operator->*(Callable&& cb) {
        return DeferStage2<Callable>{std::forward<Callable>(cb)};
    }
};

template <typename Type>
struct TempsetStage3 {
    Type& ref;
    Type t;
    Type oldval;

    explicit TempsetStage3(Type& var, std::convertible_to<Type> auto&& cv)
        : ref(var), t(std::forward<decltype(cv)>(cv)) {
        oldval = std::move(ref);
        ref = std::move(t);
    }

    ~TempsetStage3() { ref = std::move(oldval); }
};

template <typename Type>
struct TempsetStage2 {
    Type& ref;

    TempsetStage2(Type& var)
        : ref(var) {}
    TempsetStage3<Type> operator=(std::convertible_to<Type> auto&& value) {
        return TempsetStage3<Type>{ref, std::forward<decltype(value)>(value)};
    }
};

struct TempsetStage1 {
    template <typename Type>
    TempsetStage2<Type> operator->*(Type& var) {
        return TempsetStage2<Type>{var};
    }
};

} // namespace detail

template <typename T, typename... Us>
concept is_same = (std::is_same_v<T, Us> or ...);

/// More rarely used functions go here so as to not pollute
/// the global namespace too much.
namespace utils {
template <typename>
concept always_false = false;

/// ANSI Terminal colours.
enum struct Colour {
    Bold,
    Reset,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    None,
};

template <typename... Args>
struct FStringWithSrcLocImpl {
    fmt::format_string<Args...> fmt;
    std::source_location sloc;

    consteval FStringWithSrcLocImpl(
        std::convertible_to<std::string_view> auto fmt,
        std::source_location sloc = std::source_location::current()
    ) : fmt(fmt), sloc(sloc) {}
};

/// Inhibit template argument deduction.
template <typename... Args>
using FStringWithSrcLoc = FStringWithSrcLocImpl<std::type_identity_t<Args>...>;

/// RAII helper to toggle colours when printing.
///
/// Example:
/// \code{.cpp}
///     using enum Colour;
///     Colours C{true};
///     out += C(Red);
///     out += fmt::format("{}foo{}", C(Green), C(Reset));
/// \endcode
struct Colours {
    bool use_colours;
    constexpr Colours(bool use_colours)
        : use_colours{use_colours} {}

    constexpr auto operator()(Colour c) -> std::string_view {
        if (not use_colours) return "";
        switch (c) {
            case Colour::Reset: return "\033[m";
            case Colour::None: return "";
            case Colour::Red: return "\033[31m";
            case Colour::Green: return "\033[32m";
            case Colour::Yellow: return "\033[33m";
            case Colour::Blue: return "\033[34m";
            case Colour::Magenta: return "\033[35m";
            case Colour::Cyan: return "\033[36m";
            case Colour::White: return "\033[37m";
            case Colour::Bold: return "\033[1m";
        }
        return "";
    }
};

/// Check if a type is equal to one of a list of types.
template <typename T, typename... Ts>
concept is = (std::is_same_v<T, Ts> or ...);

/// Helper to stop a recursion or iteration in a callback. Unscoped
/// because it’s already in a namespace.
enum IterationResult {
    StopIteration,
    ContinueIteration,
};

/// Type-erased handle to an object that facilitates deletion.
class OpaqueHandle {
    void* ptr{};
    void (*deleter)(void*){};

public:
    OpaqueHandle(const OpaqueHandle&) = delete;
    OpaqueHandle& operator=(const OpaqueHandle&) = delete;

    OpaqueHandle(OpaqueHandle&& other) noexcept
        : ptr{std::exchange(other.ptr, nullptr)},
          deleter{std::exchange(other.deleter, nullptr)} {}

    OpaqueHandle& operator=(OpaqueHandle&& other) noexcept {
        if (this != &other) return *this;
        ptr = std::exchange(other.ptr, nullptr);
        deleter = std::exchange(other.deleter, nullptr);
        return *this;
    }

    ~OpaqueHandle() {
        if (ptr) deleter(ptr);
    }

    OpaqueHandle() = default;

    template <typename T>
    explicit OpaqueHandle(T* ptr)
        : ptr{ptr},
          deleter{[](void* ptr) { delete static_cast<T*>(ptr); }} {}
};

/// Overloaded function idiom.
template <typename... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

template <typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

/// Get the padding required to align a value to a given alignment.
template <typename T = usz>
constexpr T AlignPadding(T value, T align) {
    Assert(align != 0);
    return (align - (value % align)) % align;
}

/// Align a value to a given alignment.
template <typename T = usz>
constexpr T AlignTo(T value, T align) {
    return value + AlignPadding(value, align);
}

/// Used to implement Class::operator new(size_t, T).
template <typename Class>
auto AllocateAndRegister(
    usz sz,
    auto& owner
) -> Class* {
    static_assert(alignof(Class) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
    auto ptr = static_cast<Class*>(__builtin_operator_new(sz));
    owner.push_back(ptr);
    return ptr;
}

/// Append a range to another.
template <typename Dest, typename... Src>
void append(Dest& dest, Src&&... src) {
    (
        FWD(dest).insert(
            FWD(dest).end(),
            FWD(src).begin(),
            FWD(src).end()
        ),
        ...
    );
}

/// std::ranges::contains.
template <typename Range, typename Element, typename Proj = std::identity>
[[nodiscard]] constexpr bool contains(
    Range&& r,
    const Element& el,
    Proj&& proj = {}
) {
    return rgs::find_if(FWD(r), [&](auto&& x) { return std::invoke(proj, x) == el; }) != r.end();
}

/// Compress data and append it to a vector.
///
/// Any data already present in the vector is retained.
void Compress(
    SmallVectorImpl<u8>& into,
    ArrayRef<u8> data,
    int compression_level = 12
);

/// Used to delete an object allocated by AllocateAndRegister.
template <typename Class>
void Deallocate(Class* ptr) {
    static_assert(alignof(Class) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
    if (not ptr) return;
    __builtin_operator_delete(ptr);
}

/// Decompress data and append it to a vector.
void Decompress(
    SmallVectorImpl<u8>& into,
    ArrayRef<u8> data,
    usz uncompressed_size
);

/// Escape non-printable characters in a string.
auto Escape(StringRef str) -> std::string;

/// Compute the maximum value of an n-bit integer.
constexpr usz MaxBitValue(usz bits) {
    /// Example for 8 bits:
    ///
    /// 0000'0001 | 1
    /// 1000'0000 | b   := << bits - 1 [== (a) << 7]
    /// 0111'1111 | (b - 1)
    /// 1111'1111 | res := b | (b - 1)
    ///
    /// Note: the fastest way of doing this on x86_64
    /// is actually this `-1zu >> -bits`, but that’s
    /// *super* illegal to write in C++; fortunately,
    /// the compiler can figure out what we’re doing
    /// here and will generate the same code.
    auto b = 1zu << (bits - 1zu);
    return b | (b - 1zu);
}

/// Determine the width of a number.
auto NumberWidth(usz number, usz base = 10) -> usz;

/// Replace all occurrences of `from` with `to` in `str`.
void ReplaceAll(
    std::string& str,
    std::string_view from,
    std::string_view to
);

/// std::ranges::starts_with.
template <typename Range, typename Prefix, typename Proj = std::identity>
[[nodiscard]] constexpr bool starts_with(
    Range&& r,
    const Prefix& prefix,
    Proj&& proj = {}
) {
    return rgs::equal(
        FWD(r),
        prefix,
        [&](auto&& x, auto&& y) { return std::invoke(proj, FWD(x)) == FWD(y); }
    );
}

/// Get the smallest unique value in a range.
///
/// \param r The range to search.
/// \param filter A filter to apply to determine whether an element should be included.
/// \param proj A projection to apply to each element before comparison.
/// \return An iterator to the smallest unique element, or `r.end()` if no such element exists.
template <rgs::range Range, typename Filter, typename Proj = std::identity>
auto UniqueMin(
    Range&& r,
    Filter filter,
    Proj proj = {}
) -> std::conditional_t< //
    std::is_rvalue_reference_v<Range>,
    rgs::dangling,
    decltype(r.begin())> //
{
    if constexpr (std::is_rvalue_reference_v<Range>) return rgs::dangling{};
    else {
        /// Find the first valid element.
        auto min_el = r.begin();
        while (min_el != r.end() and not std::invoke(filter, *min_el)) ++min_el;
        auto min_it = min_el;

        /// Bail out early on an empty range.
        if (min_it == r.end()) return r.end();

        for (auto it = std::next(min_el), end = r.end(); it != end; ++it) {
            if (not std::invoke(filter, *it)) continue;
            auto&& value = std::invoke(proj, *it);
            auto&& min = std::invoke(proj, *min_el);

            /// Smaller than the minimum. This means we have a new unique minimum.
            if (value < min) min_it = min_el = it;

            /// Same as minimum value. This means we have a duplicate.
            else if (value == min) min_it = r.end();
        }

        return min_it;
    }
}
} // namespace utils

template <typename T>
requires std::is_enum_v<T>
constexpr auto operator+(T val) -> std::underlying_type_t<T> {
    return std::to_underlying(val);
}

class Context;
class File;
class TokenStream;

/// Used to represent the size of a type.
///
/// This is just a wrapper around an integer, but it requires us
/// to be explicit as to whether we want bits or bytes, which is
/// useful for avoiding mistakes.
class Size {
    usz raw;

    static_assert(CHAR_BIT == 8);
    constexpr explicit Size(usz raw) : raw{raw} {}

public:
    constexpr Size() : raw{0} {}

    [[nodiscard]] static constexpr Size Bits(usz bits) { return Size{bits}; }
    [[nodiscard]] static constexpr Size Bytes(usz bytes) { return Size{bytes * 8}; }

    /// Use of `align.value()` is necessary here because we use bits, not bytes.
    [[nodiscard]] Size align_to(Align align) const { return Size::Bytes(utils::AlignTo(bytes(), align.value())); }

    /// Get the padding required to align to a given size.
    [[nodiscard]] Size align_padding(Align align) const { return Size::Bytes(utils::AlignPadding(bytes(), align.value())); }

    [[nodiscard]] constexpr Size align_to(Size align) const { return Size{utils::AlignTo(raw, align.raw)}; }
    [[nodiscard]] constexpr auto bits() const -> usz { return raw; }
    [[nodiscard]] constexpr auto bytes() const -> usz { return utils::AlignTo<usz>(raw, 8) / 8; }

    constexpr Size operator+=(Size rhs) { return Size{raw += rhs.raw}; }
    constexpr Size operator-=(Size rhs) { return Size{raw -= rhs.raw}; }
    constexpr Size operator*=(usz rhs) { return Size{raw *= rhs}; }

    /// Only provided for Size*Integer since that basically means scaling a size. Multiplying
    /// two sizes w/ one another doesn’t make sense, so that operation is not provided.
    [[nodiscard]] friend constexpr Size operator*(Size lhs, usz rhs) { return Size{lhs.raw * rhs}; }

    [[nodiscard]] friend constexpr Size operator+(Size lhs, Size rhs) { return Size{lhs.raw + rhs.raw}; }
    [[nodiscard]] friend constexpr Size operator-(Size lhs, Size rhs) { return Size{lhs.raw - rhs.raw}; }
    [[nodiscard]] friend constexpr bool operator==(Size lhs, Size rhs) = default;
    [[nodiscard]] friend constexpr auto operator<=>(Size lhs, Size rhs) = default;
};

/// A zero-terminated string that is saved somewhere.
///
/// This is used for strings that are guaranteed to ‘live long
/// enough’ to be passed around without having to worry about who
/// owns them. This typically means they are stored in a module
/// or static storage.
class String {
    friend TokenStream;
    StringRef val;

    constexpr explicit String(StringRef val) : val{val} {}

public:
    constexpr String() = default;

    /// Construct from a string literal.
    template <usz size>
    consteval String(const char (&arr)[size]) : val{arr} {
        Assert(arr[size - 1] == '\0', "Strings must be null-terminated!");
    }

    /// Construct from a string literal.
    consteval String(llvm::StringLiteral lit) : val{lit} {}

    /// Get an iterator to the beginning of the string.
    [[nodiscard]] constexpr auto begin() const { return val.begin(); }

    /// Get the data of the string.
    [[nodiscard]] constexpr auto data() const -> const char* { return val.data(); }

    /// Check if the string is empty.
    [[nodiscard]] constexpr auto empty() const -> bool { return val.empty(); }

    /// Get an iterator to the end of the string.
    [[nodiscard]] constexpr auto end() const { return val.end(); }

    /// Check if the string ends with a given suffix.
    [[nodiscard]] constexpr auto ends_with(StringRef suffix) const -> bool {
        return val.ends_with(suffix);
    }

    /// Get the size of the string.
    [[nodiscard]] constexpr auto size() const -> usz { return val.size(); }

    /// Check if the string starts with a given prefix.
    [[nodiscard]] constexpr auto starts_with(StringRef prefix) const -> bool {
        return val.starts_with(prefix);
    }

    /// Get the string value as a std::string_view.
    [[nodiscard]] constexpr auto sv() const -> std::string_view { return val; }

    /// Get the string value.
    [[nodiscard]] constexpr auto value() const -> StringRef { return val; }

    /// Get the string value, including the null terminator.
    [[nodiscard]] constexpr auto value_with_null() const -> StringRef {
        return StringRef{val.data(), val.size() + 1};
    }

    /// Get a character at a given index.
    [[nodiscard]] constexpr auto operator[](usz idx) const -> char { return val[idx]; }

    /// Comparison operators.
    [[nodiscard]] friend auto operator==(String a, StringRef b) { return a.val == b; }
    [[nodiscard]] friend auto operator==(String a, String b) { return a.sv() == b.sv(); }
    [[nodiscard]] friend auto operator==(String a, const char* b) { return a.sv() == b; }
    [[nodiscard]] friend auto operator<=>(String a, String b) { return a.sv() <=> b.sv(); }
    [[nodiscard]] friend auto operator<=>(String a, std::string_view b) { return a.val <=> b; }

    /// Get the string.
    [[nodiscard]] constexpr operator StringRef() const { return val; }
};

/// Visit with a better order of arguments.
template <typename Variant, typename... Visitors>
void visit(Variant&& v, Visitors&&... visitors) {
    std::visit(
        utils::overloaded{std::forward<Visitors>(visitors)...},
        std::forward<Variant>(v)
    );
}
} // namespace src

namespace src::utils {
template <typename T>
concept string_like = utils::is<T, StringRef, std::string_view, String>;
}

template <>
struct fmt::formatter<llvm::StringRef> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(llvm::StringRef s, FormatContext& ctx) {
        return formatter<std::string_view>::format(std::string_view{s.data(), s.size()}, ctx);
    }
};

template <>
struct fmt::formatter<src::String> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(src::String s, FormatContext& ctx) {
        return formatter<std::string_view>::format(std::string_view{s.data(), s.size()}, ctx);
    }
};

template <>
struct fmt::formatter<src::Size> : formatter<src::u64> {
    template <typename FormatContext>
    auto format(const src::Size& sz, FormatContext& ctx) {
        return formatter<src::u64>::format(sz.bits(), ctx);
    }
};

template <>
struct fmt::formatter<llvm::APInt> : formatter<std::string_view> {
    template <typename FormatContext>
    auto format(const llvm::APInt& i, FormatContext& ctx) {
        llvm::SmallVector<char> buf;
        i.toStringSigned(buf, 10);
        return formatter<std::string_view>::format(std::string_view{buf.data(), buf.size()}, ctx);
    }
};

#if !defined(__cpp_lib_forward_like)
namespace std {
/// Clang treats this as a builtin, so we don’t actually need to implement it.
template <typename, typename T> constexpr T&& forward_like(T&& val);
}
#endif

#endif // SOURCE_MLIR_UTILS_HH
