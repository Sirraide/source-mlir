#ifndef SOURCE_MLIR_UTILS_HH
#define SOURCE_MLIR_UTILS_HH

#include <algorithm>
#include <array>
#include <chrono>
#include <coroutine>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fmt/color.h>
#include <fmt/format.h>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <memory>
#include <mlir/Support/LogicalResult.h>
#include <new>
#include <numeric>
#include <optional>
#include <ranges>
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

using llvm::cast;
using llvm::dyn_cast;
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

#define readonly_decl(type, name) \
    type _##name();               \
    __declspec(property(get = _##name)) type name

#define property_r(type, name)                                \
private:                                                      \
    type name##_field{};                                      \
public:                                                       \
    decltype(auto) _##name() const { return (name##_field); } \
    decltype(auto) _##name() { return (name##_field); }       \
    __declspec(property(get = _##name)) type name;            \
private:

#define property_rw(type, name)                                   \
private:                                                          \
    type name##_field{};                                          \
public:                                                           \
    void _##name(type value) { name##_field = std::move(value); } \
    decltype(auto) _##name() const { return name##_field; }       \
    decltype(auto) _##name() { return name##_field; }             \
    __declspec(property(get = _##name, put = _##name)) type name; \
private:

// clang-format off
#define Assert(cond, ...) (cond ? void(0) :                    \
    ::src::detail::AssertFail(                                 \
        fmt::format(                                           \
            "Assertion failed: \"" #cond "\" in {} at line {}" \
            __VA_OPT__(".\nMessage: {}"), __FILE__, __LINE__   \
            __VA_OPT__(, fmt::format(__VA_ARGS__))             \
        )                                                      \
    )                                                          \
)

#define Unreachable(...)                                       \
    ::src::detail::AssertFail(                                 \
        fmt::format(                                           \
            "Unreachable code reached in {} at line {}"        \
            __VA_OPT__(".\nMessage: {}"), __FILE__, __LINE__   \
            __VA_OPT__(, fmt::format(__VA_ARGS__))             \
        )                                                      \
    ) // clang-format on

namespace detail {
[[noreturn]] void AssertFail(std::string&& msg);

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

/// More rarely used functions go here so as to not pollute
/// the global namespace too much.
namespace utils {
/// ANSI Terminal colours.
enum struct Colour {
    Reset = 0,
    Red = 31,
    Green = 32,
    Yellow = 33,
    Blue = 34,
    Magenta = 35,
    Cyan = 36,
    White = 37,
};

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
    Colours(bool use_colours)
        : use_colours{use_colours} {}

    auto operator()(Colour c) -> std::string_view {
        if (not use_colours) return "";
        switch (c) {
            case Colour::Reset: return "\033[m";
            case Colour::Red: return "\033[31m";
            case Colour::Green: return "\033[32m";
            case Colour::Yellow: return "\033[33m";
            case Colour::Blue: return "\033[34m";
            case Colour::Magenta: return "\033[35m";
            case Colour::Cyan: return "\033[36m";
            case Colour::White: return "\033[37m";
        }
        return "";
    }
};

/// Align a value to a given alignment.
template <typename T = usz>
constexpr T AlignTo(T value, T align) {
    Assert(align != 0);
    const auto padding = (align - (value % align)) % align;
    return value + padding;
}

/// Used to implement Class::operator new(size_t, T).
template <typename Class, usz Alignment = alignof(Class)>
auto AllocateAndRegister(usz sz, auto& owner) -> void* {
    auto ptr = __builtin_operator_new(sz, std::align_val_t{Alignment});
    owner.push_back(static_cast<Class*>(ptr));
    return ptr;
}

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
} // namespace utils

} // namespace src

#endif // SOURCE_MLIR_UTILS_HH
