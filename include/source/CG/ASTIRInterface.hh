#ifndef SOURCE_ASTIRINTERFACE_HH
#define SOURCE_ASTIRINTERFACE_HH

#include <mlir/IR/Value.h>
#include <source/Frontend/AST.hh>

static_assert(src::AlignOfMLIRType >= alignof(mlir::Type));
static_assert(src::AlignOfMLIRValue >= alignof(mlir::Value));
static_assert(src::SizeOfMLIRType == sizeof(mlir::Type));
static_assert(src::SizeOfMLIRValue == sizeof(mlir::Value));

inline auto src::Expr::_mlir() -> mlir::Value {
    return std::bit_cast<mlir::Value>(_mlir_);
}

inline void src::Expr::_set_mlir(mlir::Value val) {
    std::memcpy(&_mlir_, &val, sizeof(_mlir_));
}

inline auto src::ProcDecl::_captured_locals_ptr() -> mlir::Value {
    return std::bit_cast<mlir::Value>(_captured_locals_ptr_);
}

inline void src::ProcDecl::_set_captured_locals_ptr(mlir::Value val) {
    std::memcpy(&_captured_locals_ptr_, &val, sizeof(_captured_locals_ptr_));
}

inline auto src::StructType::_mlir() -> mlir::Type {
    return std::bit_cast<mlir::Type>(_mlir_);
}

inline void src::StructType::_set_mlir(mlir::Type val) {
    std::memcpy(&_mlir_, &val, sizeof(_mlir_));
}

#endif // SOURCE_ASTIRINTERFACE_HH
