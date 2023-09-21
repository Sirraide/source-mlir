#ifndef SOURCE_FRONTEND_SEMA_HH
#define SOURCE_FRONTEND_SEMA_HH

#include <source/Frontend/AST.hh>

namespace src {
template <typename T>
struct make_formattable {
    using type = T;
};

template <typename T>
requires std::is_reference_v<T>
struct make_formattable<T> {
    using type = make_formattable<std::remove_reference_t<T>>::type;
};

template <std::derived_from<Expr> T>
struct make_formattable<T*> {
    using type = std::string;
};

template <>
struct make_formattable<Expr::TypeHandle> {
    using type = std::string;
};

template <typename T>
using make_formattable_t = make_formattable<T>::type;

class Sema {
    Module* mod{};
    ProcDecl* curr_proc{};
    Scope* curr_scope{};

public:
    /// Use Context::has_error to check for errors.
    static void Analyse(Module* mod) {
        Sema s{mod};
        s.AnalyseModule();
    }

private:
    Sema(Module* mod) : mod(mod) {}

    bool Analyse(Expr*& e);

    /// Analyse the given expression and issue an error if it is not a type.
    bool AnalyseAsType(Expr*& e);

    void AnalyseProcedure(ProcDecl* proc);
    void AnalyseModule();

    /// Convert an expression to a type, inserting implicit conversions as needed.
    bool Convert(Expr*& e, Expr* to);

    /// Returns false for convenience.
    template <typename... Args>
    bool Error(Location loc, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        Diag::Error(mod->context, loc, fmt, MakeFormattable(std::forward<Args>(args))...);
        return false;
    }

    /// Returns false for convenience.
    template <typename... Args>
    bool Error(Expr* e, fmt::format_string<make_formattable_t<Args>...> fmt, Args&&... args) {
        if (e->sema.errored) return false;
        Diag::Error(mod->context, e->location, fmt, MakeFormattable(std::forward<Args>(args))...);
        e->sema.set_errored();
        return false;
    }

    void InsertImplicitCast(Expr*& e, Expr* to);
    void InsertImplicitDereference(Expr*& e, isz depth);
    void InsertLValueReduction(Expr*& e);
    void InsertLValueToRValueConversion(Expr*& e);

    bool MakeDeclType(Expr*& e);

    template <typename T>
    auto MakeFormattable(T&& t) -> make_formattable_t<T> {
        using Type = std::remove_cvref_t<T>;
        if constexpr (std::is_pointer_v<Type> and std::derived_from<std::remove_pointer_t<Type>, Expr>) {
            return std::forward<T>(t)->type.str(true);
        } else if constexpr (std::is_same_v<Type, Expr::TypeHandle>) {
            return std::forward<T>(t).str(true);
        } else {
            return std::forward<T>(t);
        }
    }
};
} // namespace src

#endif // SOURCE_FRONTEND_SEMA_HH
