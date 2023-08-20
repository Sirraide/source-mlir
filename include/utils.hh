#ifndef SOURCE_MLIR_UTILS_HH
#define SOURCE_MLIR_UTILS_HH

#include <fmt/format.h>

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

#define STR_(X) #X
#define STR(X) STR_(X)

#define CAT_(X, Y) X##Y
#define CAT(X, Y) CAT_(X, Y)

template <typename ...arguments>
[[noreturn]] void die(fmt::format_string<arguments...> fmt, arguments&& ...args) {
    fmt::print(stderr, fmt, std::forward<arguments>(args)...);
    fmt::print(stderr, "\n");
    std::exit(1);
}

#endif // SOURCE_MLIR_UTILS_HH
