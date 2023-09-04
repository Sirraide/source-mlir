#ifndef SOURCE_INCLUDE_UTIL_STRINGTABLE_HH
#define SOURCE_INCLUDE_UTIL_STRINGTABLE_HH

#include <source/Support/Utils.hh>

namespace src {
/// String table that stores interned, null-terminated strings.
class StringTable {
    SmallVector<SmallString<32>> data;

public:
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    /// Intern a string.
    auto intern(StringRef str) -> u32 {
        auto it = rgs::find_if(data, [&](auto& s) { return s == str; });
        if (it != data.end()) return u32(it - data.begin());
        data.emplace_back(str);
        return u32(data.size() - 1);
    }
};
} // namespace src

#endif // SOURCE_INCLUDE_UTIL_STRINGTABLE_HH
