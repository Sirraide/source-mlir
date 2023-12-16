#ifndef SOURCE_SUPPORT_GENERATOR_HH
#define SOURCE_SUPPORT_GENERATOR_HH

#include <source/Support/Utils.hh>

namespace src::utils {
template <typename T>
concept GeneratableValue = std::is_move_assignable_v<T> and std::is_default_constructible_v<T>;

/// \brief A coroutine-based generator.
///
/// This can be used to create a generator that yields values.
///
/// Example:
/// \code
/// auto ints(int n) -> Generator<int> {
///     for (int i = 0; i < n; i++)
///         co_yield i;
/// }
///
/// for (auto i : ints(10))
///     fmt::print("{}\n", i);
/// \endcode
template <GeneratableValue T>
class Generator {
public:
    struct promise_type;

private:
    using handle_type = std::coroutine_handle<promise_type>;
    handle_type handle{};

    Generator(handle_type handle) noexcept : handle(handle) {}

public:
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    Generator(Generator&& other) noexcept
        : handle(std::exchange(other.handle, {})) {}

    Generator& operator=(Generator&& other) noexcept {
        if (this == &other) return *this;
        handle = std::exchange(other.handle, {});
        return *this;
    }

    ~Generator() {
        if (handle) handle.destroy();
    }

    struct promise_type {
        T current_value{};
        auto get_return_object() -> Generator { return {handle_type::from_promise(*this)}; }
        auto final_suspend() noexcept -> std::suspend_always { return {}; }
        auto initial_suspend() noexcept -> std::suspend_never { return {}; };
        void return_void() noexcept {}
        auto yield_value(std::convertible_to<T> auto&& value) -> std::suspend_always {
            current_value = std::forward<decltype(value)>(value);
            return {};
        }

        auto unhandled_exception() noexcept -> void {
            Unreachable("Unhandled exception in coroutine");
        }
    };

    class Iterator {
        friend Generator<T>;
        Generator& g;
        Iterator(Generator& g) noexcept : g(g) {}

    public:
        /// Get the current value.
        auto operator*() noexcept -> T& { return g.handle.promise().current_value; }

        /// Advance the iterator.
        auto operator++() -> Iterator& {
            Assert(not g.handle.done());
            g.handle.resume();
            return *this;
        }

        /// Check if we’re at the end.
        auto operator==(std::default_sentinel_t) const noexcept -> bool { return g.handle.done(); }
    };

    auto begin() -> Iterator { return Iterator{*this}; }
    auto end() -> std::default_sentinel_t { return {}; }
};
} // namespace src::utils

#endif // SOURCE_SUPPORT_GENERATOR_HH
