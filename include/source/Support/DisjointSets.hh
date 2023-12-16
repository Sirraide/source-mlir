#ifndef SOURCE_SUPPORT_DISJOINT_SETS_HH
#define SOURCE_SUPPORT_DISJOINT_SETS_HH

#include <source/Support/Buffer.hh>
#include <source/Support/Generator.hh>
#include <source/Support/Utils.hh>

namespace src::utils {
template <typename ValueType = int>
requires std::is_signed_v<ValueType>
class DisjointSets {
    using value_type = ValueType;

private:
    Buffer<value_type> storage;

public:

    DisjointSets(value_type size) : storage(usz(size), value_type(-1)) {
        Assert(value_type(size) == size, "Size {} must be positive for value type", size);
    }

    /// Get all elements in a set.
    auto elements(value_type x) -> Generator<value_type> {
        for (value_type i = 0, size = value_type(storage.size()); i < size; i++)
            if (find(value_type(i)) == x)
                co_yield value_type(i);
    }

    /// Find the representative of a set.
    auto find(value_type x) -> value_type {
        Assert(x < value_type(storage.size()));
        if (storage[x] < 0) return x;
        return storage[x] = find(storage[x]);
    }

    /// Merge two sets. Returns whether a merge was performed.
    bool unite(value_type x, value_type y) {
        Assert(x < value_type(storage.size()));
        Assert(y < value_type(storage.size()));
        x = find(x);
        y = find(y);
        if (x == y) return false;
        if (storage[x] > storage[y]) std::swap(x, y);
        storage[x] += storage[y];
        storage[y] = x;
        return true;
    }
};
} // namespace src::utils

#endif // SOURCE_SUPPORT_DISJOINT_SETS_HH
