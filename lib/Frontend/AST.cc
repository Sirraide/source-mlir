#include <source/Frontend/AST.hh>

src::VarDecl::VarDecl(
    std::string name,
    FunctionDecl* owner,
    Expr* type,
    Linkage linkage,
    Mangling mangling
) : ObjectDecl(EK_VarDecl, name, type, linkage, mangling),
    _owner(owner),
    _init(nullptr) {
}

auto src::VarDecl::Create(
    std::string name,
    Scope* scope,
    FunctionDecl* owner,
    Expr* type,
    Linkage linkage,
    Mangling mangling
) -> Result<Expr*> {
    auto v = new (auto{scope->module}) VarDecl(name, owner, type, linkage, mangling);
    return scope->declare(std::move(name), v);
}
