#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToFuncs/MathToFuncs.h>
#include <mlir/Conversion/MathToLibm/MathToLibm.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>
#include <source/CG/CodeGen.hh>
#include <source/Core.hh>
#include <source/HLIR/HLIRDialect.hh>
#include <source/HLIR/HLIRUtils.hh>
#include <source/Support/Utils.hh>

using namespace mlir;

namespace src {
namespace {
constexpr llvm::StringLiteral LibCFree = "free";
} // namespace

/// Perform an ‘in-memory cast’, i.e. store to memory and load as the other type.
auto CreateInMemoryCast(
    ConversionPatternRewriter& r,
    const LLVMTypeConverter& tc,
    Type to,
    Value val
) {
    to = tc.convertType(to);
    auto one = r.create<LLVM::ConstantOp>(val.getLoc(), tc.getIndexType(), r.getI32IntegerAttr(1));
    auto local = r.create<LLVM::AllocaOp>(val.getLoc(), LLVM::LLVMPointerType::get(r.getContext()), to, one);
    r.create<LLVM::StoreOp>(val.getLoc(), val, local);
    return r.create<LLVM::LoadOp>(val.getLoc(), to, local);
}

/// Lowering for string literals.
struct StringOpLowering : public ConversionPattern {
    explicit StringOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::StringOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value>,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto module = op->getParentOfType<ModuleOp>();
        auto str_op = cast<hlir::StringOp>(op);
        auto loc = op->getLoc();

        /// Create global string.
        OpBuilder::InsertionGuard i{rewriter};
        rewriter.setInsertionPointToStart(module.getBody());

        auto string_type = LLVM::LLVMArrayType::get(
            IntegerType::get(getContext(), 8),
            u32(str_op.getValue().size())
        );

        auto global = rewriter.create<LLVM::GlobalOp>(
            loc,
            string_type,
            true,
            LLVM::Linkage::Private,
            fmt::format(".str.data.{}", str_op.getIndex().getZExtValue()),
            str_op.getValueAttr(),
            0
        );

        rewriter.replaceOp(op, global);
        return success();
    }
};

/// Lowering for accessing the data pointer of a slice.
struct SliceDataOpLowering : public ConversionPattern {
    explicit SliceDataOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::SliceDataOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto slice = cast<hlir::SliceDataOp>(op);
        auto loc = op->getLoc();

        /// Get the data pointer.
        auto data_ptr = rewriter.create<LLVM::ExtractValueOp>(
            loc,
            getTypeConverter()->convertType(slice.getRes().getType()),
            operands[0],
            ArrayRef<i64>{0}
        );

        /// Replace the slice data op with the data pointer.
        rewriter.replaceOp(op, data_ptr);
        return success();
    }
};

/// Lowering for accessing the size of a slice.
struct SliceSizeOpLowering : public ConversionPattern {
    explicit SliceSizeOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::SliceSizeOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto loc = op->getLoc();

        /// Get the size.
        auto size = rewriter.create<LLVM::ExtractValueOp>(
            loc,
            getTypeConverter<LLVMTypeConverter>()->getIndexType(),
            operands[0],
            ArrayRef<i64>{1}
        );

        /// Replace the slice size op with the size.
        rewriter.replaceOp(op, size);
        return success();
    }
};

/// Lowering for global refs.
struct GlobalRefOpLowering : public ConversionPattern {
    explicit GlobalRefOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::GlobalRefOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value>,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto global_ref = cast<hlir::GlobalRefOp>(op);
        auto loc = op->getLoc();

        /// Get the global pointer.
        auto global_ptr = rewriter.create<LLVM::AddressOfOp>(
            loc,
            getTypeConverter()->convertType(global_ref.getRes().getType()),
            global_ref.getName().getLeafReference().getValue()
        );

        /// Replace the global ref op with the global pointer.
        rewriter.replaceOp(op, global_ptr);
        return success();
    }
};

/// Lowering for loads.
struct LoadOpLowering : public ConversionPattern {
    explicit LoadOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::LoadOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto load = cast<hlir::LoadOp>(op);
        auto loc = op->getLoc();

        /// Load the value.
        auto value = rewriter.create<LLVM::LoadOp>(
            loc,
            getTypeConverter()->convertType(load.getType().getType()),
            operands[0]
        );

        /// Replace the load op with the loaded value.
        rewriter.replaceOp(op, value);
        return success();
    }
};

/// Lowering for stores.
struct StoreOpLowering : public ConversionPattern {
    explicit StoreOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::StoreOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto store = cast<hlir::StoreOp>(op);
        auto loc = op->getLoc();
        hlir::StoreOpAdaptor adaptor(operands);

        /// Store the value.
        rewriter.create<LLVM::StoreOp>(
            loc,
            adaptor.getValue(),
            adaptor.getAddr(),
            store.getAlignment().getValue().getZExtValue()
        );

        /// Replace the store op with nothing.
        rewriter.eraseOp(op);
        return success();
    }
};

/// Lowering for literals.
struct LiteralOpLowering : public ConversionPattern {
    explicit LiteralOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::LiteralOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto literal = cast<hlir::LiteralOp>(op);
        auto loc = op->getLoc();

        /// How the literal is lowered depends on the type. Slice
        /// literals have a data pointer and a size.
        if (isa<hlir::SliceType>(literal.getValue().getType())) {
            /// Create a poison slice and insert the data pointer and size.
            auto ty = getTypeConverter()->convertType(literal.getValue().getType());
            auto s0 = rewriter.create<LLVM::UndefOp>(loc, ty);
            auto s1 = rewriter.create<LLVM::InsertValueOp>(loc, s0, operands[0], ArrayRef<i64>{0});
            auto s2 = rewriter.create<LLVM::InsertValueOp>(loc, s1, operands[1], ArrayRef<i64>{1});
            rewriter.replaceOp(op, s2);
            return success();
        }

        /// Invalid literal type.
        else {
            return failure();
        }
    }
};

/// Lowering for array decay ops.
struct ArrayDecayOpLowering : public ConversionPattern {
    explicit ArrayDecayOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::ArrayDecayOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto bitcast = cast<hlir::ArrayDecayOp>(op);
        auto loc = op->getLoc();

        /// Ref-to-array to ref-to-elem casts are GEPs.
        auto gep = rewriter.create<LLVM::GEPOp>(
            loc,
            getTypeConverter()->convertType(bitcast.getType()),
            getTypeConverter()->convertType(bitcast.getOperand().getType().getElem()),
            operands[0],
            ArrayRef{mlir::LLVM::GEPArg(0), mlir::LLVM::GEPArg(0)},
            true
        );

        rewriter.replaceOp(op, gep);
        return success();
    }
};

struct BitCastOpLowering : public ConversionPattern {
    explicit BitCastOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::BitCastOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto bitcast = cast<hlir::BitCastOp>(op);

        /// No-op.
        if (
            isa<hlir::ReferenceType, hlir::OptRefType>(bitcast.getType()) and
            isa<hlir::ReferenceType, hlir::OptRefType>(bitcast.getOperand().getType())
        ) {
            rewriter.replaceOp(op, args[0]);
        }

        /// This is mainly used for pointer casts, so this should probably not be reached.
        else {
            std::string from, to;
            llvm::raw_string_ostream from_os{from}, to_os{to};
            bitcast.getType().print(from_os);
            args[0].getType().print(to_os);
            Diag::ICE(
                "Unsupported lowering for BitCastOp from {} to {}",
                from,
                to
            );
        }

        return success();
    }
};

/// Lowering for var decls.
struct LocalOpLowering : public ConversionPattern {
    explicit LocalOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::LocalOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value>,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto loc = op->getLoc();
        auto var = cast<hlir::LocalOp>(op);

        /// We always pass the entire type to the alloca, so
        /// the ‘array size’ is always 1.
        auto one = rewriter.create<LLVM::ConstantOp>(
            loc,
            getTypeConverter<LLVMTypeConverter>()->getIndexType(),
            rewriter.getI32IntegerAttr(1)
        );

        /// Create the alloca.
        auto alloca = rewriter.create<LLVM::AllocaOp>(
            loc,
            LLVM::LLVMPointerType::get(getContext()),
            getTypeConverter()->convertType(var.getType().getElem()),
            one,
            var.getAlignment().getValue().getZExtValue()
        );

        /// Replace the local var op with the alloca.
        rewriter.replaceOp(op, alloca);
        return success();
    }
};

/// Lowering for zero-initialisation.
struct ZeroInitOpLowering : public ConversionPattern {
    explicit ZeroInitOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::ZeroinitialiserOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto tc = getTypeConverter<LLVMTypeConverter>();
        auto zero_init = cast<hlir::ZeroinitialiserOp>(op);
        auto type_size = DataLayout::closest(op).getTypeSize(zero_init.getOperand().getType().getElem());

        /// Generate a call to llvm.memset.
        auto zero = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(),
            rewriter.getI8Type(),
            rewriter.getI8IntegerAttr(0)
        );

        auto bytes = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(),
            tc->getIndexType(),
            IntegerAttr::get(
                rewriter.getI64Type(),
                i64(type_size * zero_init.getArraySize())
            )
        );

        rewriter.create<LLVM::MemsetOp>(
            op->getLoc(),
            operands[0],
            zero,
            bytes,
            false
        );

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lowering for structure geps.
struct StructGepOpLowering : public ConversionPattern {
    explicit StructGepOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::StructGEPOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto gep = cast<hlir::StructGEPOp>(op);
        auto loc = op->getLoc();
        hlir::StructGEPOpAdaptor adaptor(operands);

        /// Create the GEP.
        auto gep_op = rewriter.create<LLVM::GEPOp>(
            loc,
            getTypeConverter()->convertType(gep.getType()),
            getTypeConverter()->convertType(gep.getStructRef().getType().getElem()),
            adaptor.getStructRef(),
            ArrayRef{LLVM::GEPArg(0), LLVM::GEPArg(i32(gep.getIdx().getValue().getZExtValue()))},
            true
        );

        /// Replace the struct gep op with the GEP.
        rewriter.replaceOp(op, gep_op);
        return success();
    }
};

struct UnreachableOpLowering : public ConversionPattern {
    explicit UnreachableOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::UnreachableOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value>,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        rewriter.create<LLVM::UnreachableOp>(op->getLoc());
        rewriter.eraseOp(op);
        return success();
    }
};

/// Lowering for chain extractlocal.
struct ChainExtractLocalOpLowering : public ConversionPattern {
    explicit ChainExtractLocalOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::ChainExtractLocalOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto loc = op->getLoc();
        auto extract = cast<hlir::ChainExtractLocalOp>(op);
        hlir::ChainExtractLocalOpAdaptor adaptor(operands);

        /// This is a GEP, followed by a load.
        auto gep_op = rewriter.create<LLVM::GEPOp>(
            loc,
            getTypeConverter()->convertType(hlir::ReferenceType::get(extract.getType())),
            getTypeConverter()->convertType(extract.getStructRef().getType().getElem()),
            adaptor.getStructRef(),
            ArrayRef{LLVM::GEPArg(0), LLVM::GEPArg(i32(extract.getIdx().getValue().getZExtValue()))},
            true
        );

        /// Create the load.
        auto load = rewriter.create<LLVM::LoadOp>(
            loc,
            getTypeConverter()->convertType(extract.getType()),
            gep_op
        );

        /// Replace the extract local op with the load.
        rewriter.replaceOp(op, load);
        return success();
    }
};

struct DeleteOpLowering : public ConversionPattern {
    explicit DeleteOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::DeleteOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto loc = op->getLoc();

        /// Ensure free() is declared.
        auto ctx = getContext();
        if (auto m = op->getParentOfType<ModuleOp>(); not m.lookupSymbol<LLVM::LLVMFuncOp>(LibCFree)) {
            using namespace LLVM;
            PatternRewriter::InsertionGuard guard{rewriter};
            rewriter.setInsertionPointToStart(m.getBody());
            rewriter.create<LLVMFuncOp>(
                rewriter.getUnknownLoc(),
                LibCFree,
                LLVMFunctionType::get(LLVMVoidType::get(ctx), {LLVMPointerType::get(ctx)})
            );
        }

        /// Generate a call to free.
        rewriter.create<LLVM::CallOp>(
            loc,
            TypeRange{},
            FlatSymbolRefAttr::get(ctx, LibCFree),
            ArrayRef<Value>{args[0]}
        );

        rewriter.eraseOp(op);
        return success();
    }
};

struct MakeClosureOpLowering : public ConversionPattern {
    explicit MakeClosureOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::MakeClosureOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto loc = op->getLoc();
        auto make_closure = cast<hlir::MakeClosureOp>(op);
        auto tc = getTypeConverter();
        hlir::MakeClosureOpAdaptor adaptor(args);

        /// This involves creating a struct w/ two pointers: the
        /// function pointer and the closure data; the data pointer
        /// may be absent.
        auto lit = rewriter.create<LLVM::UndefOp>(loc, tc->convertType(make_closure.getType()));

        /// Store the function pointer.
        auto ref = rewriter.create<LLVM::AddressOfOp>(loc, LLVM::LLVMPointerType::get(getContext()), make_closure.getProcedure());
        auto lit2 = rewriter.create<LLVM::InsertValueOp>(loc, lit, ref, ArrayRef<i64>{0});

        /// Store the data pointer, if there is one.
        if (args.size() == 2) {
            auto lit3 = rewriter.create<LLVM::InsertValueOp>(loc, lit2, adaptor.getEnv(), ArrayRef<i64>{1});
            rewriter.replaceOp(op, lit3);
        } else {
            auto null = rewriter.create<LLVM::ZeroOp>(loc, LLVM::LLVMPointerType::get(getContext()));
            auto lit3 = rewriter.create<LLVM::InsertValueOp>(loc, lit2, null, ArrayRef<i64>{1});
            rewriter.replaceOp(op, lit3);
        }

        return success();
    }
};

struct NewOpLowering : public ConversionPattern {
    explicit NewOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::NewOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value>,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto loc = op->getLoc();
        auto new_op = cast<hlir::NewOp>(op);
        auto tc = getTypeConverter<LLVMTypeConverter>();

        /// This is a call to malloc.
        auto ctx = getContext();
        if (auto m = op->getParentOfType<ModuleOp>(); not m.lookupSymbol<LLVM::LLVMFuncOp>("malloc")) {
            using namespace LLVM;
            PatternRewriter::InsertionGuard guard{rewriter};
            rewriter.setInsertionPointToStart(m.getBody());
            rewriter.create<LLVMFuncOp>(
                rewriter.getUnknownLoc(),
                "malloc",
                LLVMFunctionType::get(LLVMPointerType::get(ctx), {tc->getIndexType()})
            );
        }

        auto bytes = DataLayout::closest(op).getTypeSize(new_op.getResult().getType());
        auto size = rewriter.create<LLVM::ConstantOp>(
            loc,
            tc->getIndexType(),
            rewriter.getI64IntegerAttr(i64(bytes))
        );

        auto malloc = rewriter.create<LLVM::CallOp>(
            loc,
            LLVM::LLVMPointerType::get(getContext()),
            FlatSymbolRefAttr::get(ctx, "malloc"),
            ArrayRef<Value>{size}
        );

        rewriter.replaceOp(op, malloc.getResult());
        return success();
    }
};

struct NilOpLowering : public ConversionPattern {
    explicit NilOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::NilOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value>,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto nil = cast<hlir::NilOp>(op);
        rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, getTypeConverter()->convertType(nil.getType()));
        return success();
    }
};

struct NotOpLowering : public ConversionPattern {
    explicit NotOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::NotOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto loc = op->getLoc();

        /// This is a xor with 1.
        auto one = rewriter.create<LLVM::ConstantOp>(
            loc,
            rewriter.getI1Type(),
            rewriter.getIntegerAttr(rewriter.getI1Type(), 1)
        );

        auto xor_op = rewriter.create<LLVM::XOrOp>(
            loc,
            rewriter.getI1Type(),
            args[0],
            one
        );

        rewriter.replaceOp(op, xor_op.getResult());
        return success();
    }
};

struct OffsetOpLowering : public ConversionPattern {
    explicit OffsetOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::OffsetOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto tc = getTypeConverter();
        auto offset = cast<hlir::OffsetOp>(op);
        rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
            op,
            tc->convertType(offset.getPointer().getType()),
            tc->convertType(offset.getPointer().getType().getElem()),
            args[0],
            ArrayRef{LLVM::GEPArg(args[1])},
            true
        );
        return success();
    }
};

struct PointerEqOpLowering : public ConversionPattern {
    explicit PointerEqOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::PointerEqOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        /// This is a comparison of two pointers.
        auto cmp = rewriter.create<LLVM::ICmpOp>(
            op->getLoc(),
            LLVM::ICmpPredicate::eq,
            args[0],
            args[1]
        );

        rewriter.replaceOp(op, cmp.getResult());
        return success();
    }
};

struct PointerNeOpLowering : public ConversionPattern {
    explicit PointerNeOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::PointerNeOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        /// This is a comparison of two pointers.
        auto cmp = rewriter.create<LLVM::ICmpOp>(
            op->getLoc(),
            LLVM::ICmpPredicate::ne,
            args[0],
            args[1]
        );

        rewriter.replaceOp(op, cmp.getResult());
        return success();
    }
};

struct InvokeClosureOpLowering : public ConversionPattern {
    explicit InvokeClosureOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::InvokeClosureOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& rewriter
    ) const -> LogicalResult override {
        auto loc = op->getLoc();
        auto invoke_closure = cast<hlir::InvokeClosureOp>(op);
        hlir::InvokeClosureOpAdaptor adaptor(args);

        /// Extract function pointer and env.
        auto fn_ptr = rewriter.create<LLVM::ExtractValueOp>(
            loc,
            LLVM::LLVMPointerType::get(getContext()),
            adaptor.getClosure(),
            ArrayRef<i64>{0}
        );

        auto env = rewriter.create<LLVM::ExtractValueOp>(
            loc,
            LLVM::LLVMPointerType::get(getContext()),
            adaptor.getClosure(),
            ArrayRef<i64>{1}
        );

        /// Invoke the function pointer.
        SmallVector<mlir::Value, 8> invoke_args;
        invoke_args.push_back(fn_ptr);
        for (auto a : args | vws::drop(1)) invoke_args.push_back(a);
        invoke_args.push_back(env);
        auto invoke = rewriter.create<LLVM::CallOp>(
            loc,
            getTypeConverter()->convertType(invoke_closure.getType()),
            invoke_args
        );

        rewriter.replaceOp(op, invoke.getResult());
        return success();
    }
};

struct FuncOpLowering : public ConversionPattern {
    explicit FuncOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::FuncOp::getOperationName(), 1, ctx) {
    }

    /// See also FuncOpConversionBase in FuncToLLVM.cpp.
    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value>,
        ConversionPatternRewriter& r
    ) const -> LogicalResult override {
        auto func = cast<hlir::FuncOp>(op);
        auto ftype = func.getFunctionType();
        auto tc = getTypeConverter<LLVMTypeConverter>();
        auto u = UnitAttr::get(r.getContext());

        /// Convert arguments.
        TypeConverter::SignatureConversion res{func.getNumArguments()};
        auto converted = tc->convertFunctionSignature(ftype, func.getVariadic(), true, res);
        if (not converted) return failure();

        /// Create the function.
        auto linkage = func.getLinkage().getLinkage();
        auto llvm_func = r.create<LLVM::LLVMFuncOp>(
            func->getLoc(),
            func.getName(),
            converted,
            linkage,
            false,
            func.getCc()
        );

        /// Add the appropriate parameter attributes.
        for (auto [i, t] : llvm::enumerate(ftype.getInputs())) {
            if (auto ref = dyn_cast<hlir::ReferenceType>(t)) {
                auto sz = DataLayout::closest(op).getTypeSize(ref.getElem());
                llvm_func.setArgAttr(u32(i), "llvm.dereferenceable", IntegerAttr::get(tc->getIndexType(), i64(sz)));
                llvm_func.setArgAttr(u32(i), "llvm.nonnull", u);
                llvm_func.setArgAttr(u32(i), "llvm.noundef", u);
                llvm_func.setArgAttr(u32(i), "llvm.nofree", u);
                llvm_func.setArgAttr(u32(i), "llvm.nocapture", u);
            }
        }

        /// Add the appropriate function attributes.
        llvm_func.setPassthroughAttr(r.getArrayAttr(StringAttr::get(getContext(), "nounwind")));

        /// Move the body over.
        r.inlineRegionBefore(func.getBody(), llvm_func.getBody(), llvm_func.end());
        if (failed(r.convertRegionTypes(&llvm_func.getBody(), *tc, &res))) return failure();
        r.replaceOp(func, llvm_func);
        return success();
    }
};

struct CallOpLowering : public ConversionPattern {
    explicit CallOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::CallOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> args,
        ConversionPatternRewriter& r
    ) const -> LogicalResult override {
        auto loc = op->getLoc();
        auto call = cast<hlir::CallOp>(op);
        hlir::CallOpAdaptor adaptor(args);

        /// Create the call.
        auto llvm_call = r.create<LLVM::CallOp>(
            loc,
            call.getYield() ? getTypeConverter()->convertType(call.getYield().getType()) : TypeRange{},
            call.getCallee(),
            adaptor.getArgs()
        );

        if (call.getYield()) r.replaceOp(op, llvm_call.getResult());
        else r.eraseOp(op);
        return success();
    }
};

struct ReturnOpLowering : public ConversionPattern {
    explicit ReturnOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, hlir::ReturnOp::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> arguments,
        ConversionPatternRewriter& r
    ) const -> LogicalResult override {
        /// Create the return.
        r.create<LLVM::ReturnOp>(
            op->getLoc(),
            arguments.empty() ? mlir::ValueRange{} : arguments[0]
        );

        r.eraseOp(op);
        return success();
    }
};

template <typename Op, typename ArithOp>
struct ArithOpLowering : public ConversionPattern {
    explicit ArithOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, Op::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> arguments,
        ConversionPatternRewriter& r
    ) const -> LogicalResult override {
        Op b = cast<Op>(op);

        /// If the arguments are not arrays, just lower to LLVM ops.
        if (not isa<hlir::ArrayType>(b.getRes().getType())) {
            r.replaceOpWithNewOp<ArithOp>(op, arguments[0], arguments[1]);
            return success();
        }

        /// Arrays of i8, i16 etc are easy to lower because the element alignments
        /// line up with the alignments of the corresponding vector types.
        auto arr = cast<hlir::ArrayType>(b.getRes().getType());
        auto elem = dyn_cast<IntegerType>(arr.getElem());
        Assert(elem, "We currently only support arithmetic on arrays of integers");

        /// The semantics of a vector of integer types whose bit width is not a power
        /// of two aren’t entirely clear wrt vectors vs arrays; e.g. an array of i17s
        /// is really an array of i32s, but a vector of i17s may consist of packed i17s,
        /// in which case we can’t simply treat one as the other. That should always be
        /// possible for power-of-two integers, though, so we can simply reinterpret those
        /// as vectors and let the vector dialect do the rest.
        Assert(
            llvm::isPowerOf2_64(elem.getWidth()) and elem.getWidth() >= 8,
            "We currently only support arithmetic on arrays of power-of-two integers of at least 8 bits"
        );

        /// Convert to vectors.
        auto tc = getTypeConverter<LLVMTypeConverter>();
        auto vector_type = VectorType::get(i64(arr.getSize()), elem);
        auto lhs = CreateInMemoryCast(r, *tc, vector_type, arguments[0]);
        auto rhs = CreateInMemoryCast(r, *tc, vector_type, arguments[1]);
        auto res = r.create<ArithOp>(op->getLoc(), lhs, rhs);
        auto conv = CreateInMemoryCast(r, *tc, arr, res);
        r.replaceAllUsesWith(op->getResult(0), conv);
        r.eraseOp(op);
        return success();
    }
};

template <typename Op, arith::CmpIPredicate pred>
struct CmpOpLowering : public ConversionPattern {
    explicit CmpOpLowering(MLIRContext* ctx, LLVMTypeConverter& tc)
        : ConversionPattern(tc, Op::getOperationName(), 1, ctx) {
    }

    auto matchAndRewrite(
        Operation* op,
        ArrayRef<Value> arguments,
        ConversionPatternRewriter& r
    ) const -> LogicalResult override {
        Op b = cast<Op>(op);

        /// If the arguments are not arrays, just lower to LLVM ops.
        if (not isa<hlir::ArrayType>(b.getRes().getType())) {
            r.replaceOpWithNewOp<arith::CmpIOp>(op, pred, arguments[0], arguments[1]);
            return success();
        }

        /// See ArithOpLowering for more info on this.
        auto arr = cast<hlir::ArrayType>(b.getLhs().getType());
        auto elem = dyn_cast<IntegerType>(arr.getElem());
        Assert(elem, "We currently only support arithmetic on arrays of integers");
        Assert(
            llvm::isPowerOf2_64(elem.getWidth()) and elem.getWidth() >= 8,
            "We currently only support arithmetic on arrays of power-of-two integers of at least 8 bits"
        );

        /// Convert to vectors.
        auto tc = getTypeConverter<LLVMTypeConverter>();
        auto vector_type = VectorType::get(i64(arr.getSize()), elem);
        auto lhs = CreateInMemoryCast(r, *tc, vector_type, arguments[0]);
        auto rhs = CreateInMemoryCast(r, *tc, vector_type, arguments[1]);
        auto res = r.create<arith::CmpIOp>(op->getLoc(), pred, lhs, rhs);
        auto conv = CreateInMemoryCast(r, *tc, b.getRes().getType(), res);
        r.replaceAllUsesWith(op->getResult(0), conv);
        r.eraseOp(op);
        return success();
    }
};

struct HLIRToLLVMLoweringPass
    : public PassWrapper<HLIRToLLVMLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }

    void runOnOperation() final {
        /// Nested pass manager for converting math operations. We need to
        /// perform these conversions early as there is currently no way
        /// to integrate them into the rest of the conversion pipeline. Also
        /// perform lowering to llvm intrinsics here to avoid generating
        /// unnecessary libm calls.
        OpPassManager opm{"builtin.module"};
        opm.addPass(createConvertMathToFuncs());
        opm.addNestedPass<hlir::FuncOp>(createCanonicalizerPass());
        opm.addNestedPass<hlir::FuncOp>(createConvertMathToLLVMPass());
        if (failed(runPipeline(opm, getOperation())))
            return signalPassFailure();

        /// Convert slice types to structs of ptr + index.
        LLVMConversionTarget target{getContext()};
        LLVMTypeConverter tc{&getContext()};
        RewritePatternSet patterns{&getContext()};
        tc.addConversion([&](hlir::SliceType) {
            return LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {LLVM::LLVMPointerType::get(&getContext()), tc.getIndexType()}
            );
        });

        /// Convert closures to { ptr, ptr }.
        tc.addConversion([&](hlir::ClosureType) {
            auto ptr = LLVM::LLVMPointerType::get(&getContext());
            return LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {ptr, ptr}
            );
        });

        /// Convert reference types to ptr.
        tc.addConversion([&](hlir::ReferenceType) {
            return LLVM::LLVMPointerType::get(&getContext());
        });

        /// Convert optional references to ptr.
        tc.addConversion([&](hlir::OptRefType) {
            return LLVM::LLVMPointerType::get(&getContext());
        });

        /// Convert array types to arrays.
        tc.addConversion([&](hlir::ArrayType arr) {
            auto elem = tc.convertType(arr.getElem());
            Assert(elem, "Array type has invalid element type", (arr.dump(), 0));
            return LLVM::LLVMArrayType::get(elem, unsigned(arr.getSize()));
        });

        /// Convert none to void.
        tc.addConversion([&](NoneType) {
            return LLVM::LLVMVoidType::get(&getContext());
        });

        populateAffineToStdConversionPatterns(patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        arith::populateArithToLLVMConversionPatterns(tc, patterns);
        populateFinalizeMemRefToLLVMConversionPatterns(tc, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(tc, patterns);
        populateFuncToLLVMConversionPatterns(tc, patterns);
        populateFuncToLLVMFuncOpConversionPattern(tc, patterns);
        index::populateIndexToLLVMConversionPatterns(tc, patterns);
        populateMathToLLVMConversionPatterns(tc, patterns);
        populateMathToLibmConversionPatterns(patterns);

        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalOp<ModuleOp>();

        // clang-format off
        patterns.add<
            ArithOpLowering<hlir::AddOp, arith::AddIOp>,
            ArithOpLowering<hlir::AndOp, arith::AndIOp>,
            ArithOpLowering<hlir::DivOp, arith::DivSIOp>,
            ArithOpLowering<hlir::MulOp, arith::MulIOp>,
            ArithOpLowering<hlir::OrOp, arith::OrIOp>,
            ArithOpLowering<hlir::RemOp, arith::RemSIOp>,
            ArithOpLowering<hlir::SarOp, arith::ShRSIOp>,
            ArithOpLowering<hlir::ShlOp, arith::ShLIOp>,
            ArithOpLowering<hlir::ShrOp, arith::ShRUIOp>,
            ArithOpLowering<hlir::SubOp, arith::SubIOp>,
            ArithOpLowering<hlir::XorOp, arith::XOrIOp>,
            ArrayDecayOpLowering,
            BitCastOpLowering,
            CallOpLowering,
            ChainExtractLocalOpLowering,
            CmpOpLowering<hlir::EqOp, arith::CmpIPredicate::eq>,
            CmpOpLowering<hlir::NeOp, arith::CmpIPredicate::ne>,
            CmpOpLowering<hlir::LtOp, arith::CmpIPredicate::slt>,
            CmpOpLowering<hlir::LeOp, arith::CmpIPredicate::sle>,
            CmpOpLowering<hlir::GtOp, arith::CmpIPredicate::sgt>,
            CmpOpLowering<hlir::GeOp, arith::CmpIPredicate::sge>,
            DeleteOpLowering,
            FuncOpLowering,
            GlobalRefOpLowering,
            InvokeClosureOpLowering,
            LiteralOpLowering,
            LoadOpLowering,
            LocalOpLowering,
            MakeClosureOpLowering,
            NewOpLowering,
            NilOpLowering,
            NotOpLowering,
            OffsetOpLowering,
            PointerEqOpLowering,
            PointerNeOpLowering,
            ReturnOpLowering,
            SliceDataOpLowering,
            SliceSizeOpLowering,
            StoreOpLowering,
            StringOpLowering,
            StructGepOpLowering,
            UnreachableOpLowering,
            ZeroInitOpLowering
        >(&getContext(), tc);
        // clang-format on

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            return signalPassFailure();
    }
};
} // namespace src

void src::LowerToLLVM(mlir::MLIRContext* ctx, Module* mod, bool debug_llvm_lowering, bool no_verify) {
    /// Lower the module.
    if (debug_llvm_lowering) ctx->disableMultithreading();
    mlir::PassManager pm{ctx};
    if (no_verify) pm.enableVerifier(false);
    pm.addPass(std::make_unique<HLIRToLLVMLoweringPass>());
    if (debug_llvm_lowering) pm.enableIRPrinting();
    if (mlir::failed(pm.run(mod->mlir)))
        Diag::ICE(mod->context, mod->module_decl_location, "Module lowering failed");
}

void src::GenerateLLVMIR(Module* mod, int opt_level, llvm::TargetMachine* machine) {
    if (mod->llvm) return;
    mod->llvm = mlir::translateModuleToLLVMIR(
        mod->mlir,
        mod->llvm_context,
        mod->is_logical_module ? mod->name : "Source Executable"
    );

    /// Optimise the module, if requested.
    auto xfrm = mlir::makeOptimizingTransformer(unsigned(std::min(opt_level, 3)), 0, machine);
    if (auto res = xfrm(mod->llvm.get()))
        Diag::ICE(mod->context, {}, "Failed to optimise Module: {}", llvm::toString(std::move(res)));

    /// Write module description.
    if (mod->is_logical_module) {
        auto md = mod->serialise();
        auto nm = mod->description_section_name();
        auto ty = llvm::ArrayType::get(llvm::IntegerType::getInt8Ty(mod->llvm->getContext()), md.size());
        auto cst = llvm::ConstantDataArray::get(mod->llvm_context, md);
        mod->llvm->getOrInsertGlobal(
            nm,
            ty,
            [&] {
                auto var = new llvm::GlobalVariable(
                    *mod->llvm,
                    ty,
                    true,
                    llvm::GlobalValue::PrivateLinkage,
                    cst,
                    nm
                );

                var->setSection(nm);
                var->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
                var->setVisibility(llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
                return var;
            }
        );
    }
}

void src::Module::print_llvm(int opt_level) {
    GenerateLLVMIR(this, opt_level, nullptr);
    llvm->print(llvm::outs(), nullptr);
}

int src::Module::run(int opt_level) {
    Assert(not is_logical_module, "Module is not executable");
    Assert(mlir, "Must codegen module before executing");

    /// Create optimiser.
    auto engine = mlir::ExecutionEngine::create(
        mlir,
        {
            .jitCodeGenOptLevel = llvm::CodeGenOptLevel(std::clamp(opt_level, 0, 3)),
        }
    );
    Assert(engine, "Failed to create execution engine");

    /// Invoke __src_main.
    auto result = engine.get()->invokePacked("__src_main");
    if (result) {
        Diag::Error(context, {}, "Execution failed: {}", llvm::toString(std::move(result)));
        return 1;
    }
    return 0;
}
