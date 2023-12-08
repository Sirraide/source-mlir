//
// Created by ae on 08/12/23.
//

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
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

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
    requires (not std::is_same_v<std::remove_cvref_t<Range>, Buffer<T>>)
    explicit Buffer(Range&& range) : Buffer{std::begin(range), std::end(range)} {}

    /// Get an iterator to the start of the buffer.
    [[nodiscard]] auto begin() const -> const_iterator { return buffer.get(); }
    [[nodiscard]] auto begin() -> iterator { return buffer.get(); }

    /// Get a pointer to the buffer data.
    [[nodiscard]] auto data() const -> const T* { return buffer.get(); }
    [[nodiscard]] auto data() -> T* { return buffer.get(); }

    /// Check if the buffer is empty.
    [[nodiscard]] auto empty() const -> bool { return size() == 0; }

    /// Get an iterator to the end of the buffer.
    [[nodiscard]] auto end() const -> const_iterator { return buffer.get() + size(); }
    [[nodiscard]] auto end() -> iterator { return buffer.get() + size(); }

    /// Get the buffer size.
    [[nodiscard]] auto size() const -> usz { return element_count; }

    /// Access an element of the buffer.
    [[nodiscard]] auto operator[](usz idx) const -> const_reference {
        CheckInbounds(idx);
        return buffer[idx];
    }

    /// Access an element of the buffer.
    [[nodiscard]] auto operator[](usz idx) -> reference {
        CheckInbounds(idx);
        return buffer[idx];
    }

    /// Access a range of elements of the buffer.
    [[nodiscard]] auto operator[](usz start, usz end) const -> std::span<const T> {
        CheckInbounds(start);
        CheckInbounds(end);
        return std::span{buffer.get() + start, end - start};
    }

    /// Access a range of elements of the buffer.
    [[nodiscard]] auto operator[](usz start, usz end) -> std::span<T> {
        CheckInbounds(start);
        CheckInbounds(end);
        return std::span{buffer.get() + start, end - start};
    }

    operator ArrayRef<T>() const { return {buffer.get(), size()}; }
private:
    void CheckInbounds(usz idx) const {
        LCC_ASSERT(idx < size(), "Index out of bounds");
    }
};
}

#endif // SOURCE_BUFFER_HH
