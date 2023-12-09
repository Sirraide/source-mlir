#ifndef SOURCE_BUFFER_HH
#define SOURCE_BUFFER_HH

#include <source/Support/Utils.hh>

namespace src {
/// Fixed-sized buffer, intended to be used where a vector
/// could be used, but resizing is not needed.
template <typename T>
requires (not std::is_reference_v<T>, not std::is_const_v<T>)
class Buffer {
    /// The buffer.
    std::unique_ptr<T[]> buffer{};
    usz element_count{};

public:
    using value_type = T;

    /// Create an empty buffer.
    explicit Buffer() = default;

    /// Create a new buffer that can hold N elements.
    explicit Buffer(usz size)
        : buffer{std::make_unique<T[]>(size)},
          element_count{size} {}

    /// Create a new buffer that can hold N elements and initialize it with a value.
    explicit Buffer(usz size, T val)
        : Buffer{size} {
        std::fill(begin(), end(), val);
    }

    /// Create a new buffer from an iterator range.
    template <typename Iter>
    explicit Buffer(Iter begin, Iter end)
        : buffer{std::make_unique<T[]>(usz(std::distance(begin, end)))},
          element_count{usz(std::distance(begin, end))} {
        std::copy(begin, end, buffer.get());
    }

    /// Create a buffer from an ArrayRef.
    Buffer(ArrayRef<T> ref) : Buffer{ref.begin(), ref.end()} {}

    /// Create a new buffer from a range.
    template <typename Range>
    requires (not std::is_same_v<std::remove_cvref_t<Range>, Buffer>)
    explicit Buffer(Range&& range) : Buffer{std::begin(range), std::end(range)} {}

    [[nodiscard]] auto begin(this auto&& self) { return FWD(self).buffer.get(); }
    [[nodiscard]] auto data(this auto&& self) { return FWD(self).begin(); }
    [[nodiscard]] auto empty() const -> bool { return size() == 0; }
    [[nodiscard]] auto end(this auto&& self) { return FWD(self).data() + FWD(self).size(); }
    [[nodiscard]] auto size() const -> usz { return element_count; }

    /// Access an element of the buffer.
    [[nodiscard]] auto operator[](this auto&& self, usz idx) {
        FWD(self).CheckInbounds(idx);
        return FWD(self).buffer[idx];
    }

    /// Access a range of elements of the buffer.
    template <typename This>
    [[nodiscard]] auto operator[](
        this This&& self,
        usz start,
        usz end
    ) -> std::span<std::remove_cvref_t<decltype(*self.data())>> {
        self.CheckInbounds(start);
        self.CheckInbounds(end);
        return std::span{self.data() + start, end - start};
    }

    operator ArrayRef<T>() const { return {buffer.get(), size()}; }

private:
    void CheckInbounds(usz idx) const {
        Assert(idx < size(), "Index out of bounds");
    }
};
} // namespace src

#endif // SOURCE_BUFFER_HH
