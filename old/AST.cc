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