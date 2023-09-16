#ifndef SOURCE_INCLUDE_UTIL_STRINGTABLE_HH
#define SOURCE_INCLUDE_UTIL_STRINGTABLE_HH

#include <source/Support/Utils.hh>

namespace src {
/// String table that stores interned strings that are *not* null-terminated.
class StringTable {
    SmallVector<std::string, 0> data;

public:
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    /// Intern a string.
    auto intern(StringRef str) -> u32 {
        auto it = rgs::find_if(data, [&](auto& s) { return s == str; });
        if (it != data.end()) return u32(it - data.begin());
        auto s = data.emplace_back(str);
        return u32(data.size() - 1);
    }

    /// Get the string data at the given index.
    auto operator[](usz idx) const -> const char* {
        Assert(idx < data.size());
        return data[idx].data();
    }
};
} // namespace src

#endif // SOURCE_INCLUDE_UTIL_STRINGTABLE_HH
