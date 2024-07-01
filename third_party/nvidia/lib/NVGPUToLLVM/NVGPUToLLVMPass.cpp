#include "NVGPUToLLVM/NVGPUToLLVMPass.h"

#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "NVGPUToLLVM/Passes.h.inc"

namespace ttn = mlir::triton::nvgpu;
using ::mlir::LLVM::NVIDIA::getSRegValue;
using ttn::Constraints;
using ttn::OperandsAndConstraints;

namespace {

const std::string Wgmma_Fence_Op = "wgmma.fence.sync.aligned;";
const std::string Wgmma_Commit_Group_Op = "wgmma.commit_group.sync.aligned;";
const std::string Cluster_Wait_Op = "barrier.cluster.wait.aligned;";
const std::string Fence_Mbarrier_Init_Op =
    "fence.mbarrier_init.release.cluster;";
const std::string Cluster_Cta_Id_Op = "{\n"
                                      ".reg .u32 a<5>;              \n"
                                      "mov.u32 a0, %cluster_ctaid.x;\n"  // x
                                      "mov.u32 a1, %cluster_ctaid.y;\n"  // y
                                      "mov.u32 a2, %cluster_ctaid.z;\n"  // z
                                      "mov.u32 a3, %cluster_nctaid.x;\n" // nx
                                      "mov.u32 a4, %cluster_nctaid.y;\n" // ny
                                      "mad.lo.u32 a1, a2, a4, a1;     \n"
                                      "mad.lo.u32 $0, a1, a3, a0;     \n"
                                      "}";
const std::string Canonical_Warp_Id_Op =
    "{\n"
    ".reg .u32 a<5>;              \n"
    "mov.u32 a0, %tid.x;          \n" // x
    "mov.u32 a1, %tid.y;          \n" // y
    "mov.u32 a2, %tid.z;          \n" // z
    "mov.u32 a3, %ntid.x;         \n" // nx
    "mov.u32 a4, %ntid.y;         \n" // ny
    "mad.lo.u32 a1, a2, a4, a1;   \n"
    "mad.lo.u32 a0, a1, a3, a0;   \n"
    "shr.u32 a0, a0, 5;           \n"
    ".reg .b32         %tmp<3>;   \n"
    "mov.u32   %tmp0, -1;         \n"
    "mov.u32   %tmp1, 31;         \n"
    "mov.u32   %tmp2, 0;          \n"
    "shfl.sync.idx.b32         $0, a0, %tmp2, %tmp1, %tmp0;           \n"
    "}";

bool isNumber(const std::string &s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

Type getTypeFromConstraint(char constraint, PatternRewriter &rewriter) {
  Type ty;
  if (constraint == 'b')
    ty = IntegerType::get(rewriter.getContext(), 1);
  else if (constraint == 'h')
    ty = IntegerType::get(rewriter.getContext(), 16);
  else if (constraint == 'r')
    ty = IntegerType::get(rewriter.getContext(), 32);
  else if (constraint == 'l')
    ty = IntegerType::get(rewriter.getContext(), 64);
  else if (constraint == 'f')
    ty = FloatType::getF32(rewriter.getContext());
  else if (constraint == 'd')
    ty = FloatType::getF64(rewriter.getContext());
  else {
    assert(false && "Unsupported constraint");
  }
  return ty;
}

// Converts the given value to the type represented by the constraint
// E.g. if val is of type llvmptr and constraint is 'r', then we convert
// val to i32 using ptrtoint(i32_ty, val)
Value convertToType(Value val, std::string constraint, Location loc,
                    PatternRewriter &rewriter) {
  auto isConstraintNumber = isNumber(constraint);
  if (!isConstraintNumber) {
    auto ty = getTypeFromConstraint(constraint[0], rewriter);
    if (isa<LLVM::LLVMPointerType>(val.getType())) {
      return ptrtoint(ty, val);
    } else {
      assert(val.getType().getIntOrFloatBitWidth() <=
                 ty.getIntOrFloatBitWidth() &&
             "Cannot convert to a smaller type");
      if (val.getType().getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth())
        return zext(ty, val);
    }
  }
  return val;
}

SmallVector<PTXBuilder::Operand *>
getPtxOutputs(const nvgpu::Constraints &outputConstraints,
              PTXBuilder &ptxBuilder) {
  SmallVector<PTXBuilder::Operand *> ptxOutputs;
  for (unsigned i = 0; i < outputConstraints.size(); i++) {
    auto *ptxOutput = ptxBuilder.newOperand(outputConstraints[i]);
    ptxOutputs.push_back(ptxOutput);
  }
  return ptxOutputs;
}

OperandsAndConstraints
unpackOperands(const OperandsAndConstraints &operandsAndConstraints,
               PTXBuilder &ptxBuilder, Location loc,
               PatternRewriter &rewriter) {
  OperandsAndConstraints unpackedOperands;
  for (const auto &[operand, constraint] : operandsAndConstraints) {
    auto llvmStruct = llvm::dyn_cast<LLVM::LLVMStructType>(operand.getType());
    // if a constraint is a number, then we are doing input/output tying
    // if the operand is a struct, then we need to unpack it, and
    // add the constraint to each of the unpacked operands uses the constraint
    // as an offset
    auto isConstraintNumber = isNumber(constraint);
    if (llvmStruct) {
      for (unsigned i = 0; i < llvmStruct.getBody().size(); i++) {
        if (isConstraintNumber) {
          auto constraintInt = std::stoi(constraint) + i;
          unpackedOperands.push_back(
              {extract_val(llvmStruct.getBody()[i], operand, i),
               std::to_string(constraintInt)});
        } else {
          unpackedOperands.push_back(
              {extract_val(llvmStruct.getBody()[i], operand, i), constraint});
        }
      }
    } else {
      unpackedOperands.push_back({operand, constraint});
    }
  }
  return unpackedOperands;
}

SmallVector<PTXBuilder::Operand *>
getPtxOperands(const OperandsAndConstraints &operandsAndConstraints,
               PTXBuilder &ptxBuilder, Location loc,
               PatternRewriter &rewriter) {
  SmallVector<PTXBuilder::Operand *> ptxOperands;
  auto unpackedOperandsAndConstraints =
      unpackOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
  for (auto &[operand, constraint] : unpackedOperandsAndConstraints) {
    auto convertedOperand = convertToType(operand, constraint, loc, rewriter);
    auto *ptxOperand = ptxBuilder.newOperand(convertedOperand, constraint);
    ptxOperands.push_back(ptxOperand);
  }
  return ptxOperands;
}

std::string patchPtxAsm(Operation *op, std::string ptxAsm) {
  std::vector<std::pair<int, int>> patchLocations;
  std::vector<std::string> patchValues;
  auto start = ptxAsm.find("#", 0);
  while (start != std::string::npos) {
    auto endIterator =
        std::find_if(ptxAsm.begin() + start + 1, ptxAsm.end(),
                     [](unsigned char c) { return !std::isalnum(c); });

    assert(endIterator != ptxAsm.end() && "unexpected asm format");

    auto end = std::distance(ptxAsm.begin(), endIterator);
    auto patchLocation = std::make_pair(start, end);
    patchLocations.push_back(patchLocation);
    auto patchValue = ptxAsm.substr(start + 1, end - start - 1);
    patchValues.push_back(patchValue);
    start = ptxAsm.find("#", end);
  }
  assert(patchLocations.size() == patchValues.size() &&
         "patchLocations and patchValues should have the same size");
  if (patchLocations.size() == 0) {
    return ptxAsm;
  }
  std::string res = "";
  size_t prevStart = 0;
  unsigned i = 0;
  for (auto &[start, end] : patchLocations) {
    res += ptxAsm.substr(prevStart, start - prevStart);
    auto integerAttr = op->getAttrOfType<IntegerAttr>(patchValues[i]);
    auto attr = integerAttr.getInt();
    res += std::to_string(attr);
    prevStart = end;
    i++;
  }
  if (prevStart < ptxAsm.size())
    res += ptxAsm.substr(prevStart, ptxAsm.size() - prevStart);
  return res;
}

template <typename SourceOp>
class NVGPUOpGenericPattern : public OpRewritePattern<SourceOp> {
public:
  explicit NVGPUOpGenericPattern(MLIRContext *context, std::string ptxAsm,
                                 Constraints outputConstraints,
                                 Constraints inputConstraints)
      : OpRewritePattern<SourceOp>(context), ptxAsm(std::move(ptxAsm)),
        outputConstraints(outputConstraints),
        inputConstraints(inputConstraints) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
      operandsAndConstraints.push_back(
          {op->getOperand(i), inputConstraints[i]});
    }
    return rewriteAsPtxAsm(op, rewriter, ptxAsm, operandsAndConstraints,
                           outputConstraints);
  }

private:
  std::string ptxAsm;
  Constraints outputConstraints;
  Constraints inputConstraints;
};

class FenceAsyncSharedOpPattern
    : public OpRewritePattern<ttn::FenceAsyncSharedOp> {
public:
  using OpRewritePattern<ttn::FenceAsyncSharedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::FenceAsyncSharedOp op,
                                PatternRewriter &rewriter) const override {
    std::string ptxAsm = op.getBCluster() ? "fence.proxy.async.shared::cluster;"
                                          : "fence.proxy.async.shared::cta;";
    return rewriteAsPtxAsm(op, rewriter, std::move(ptxAsm));
  }
};

class ClusterArriveOpPattern : public OpRewritePattern<ttn::ClusterArriveOp> {
public:
  using OpRewritePattern<ttn::ClusterArriveOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::ClusterArriveOp op,
                                PatternRewriter &rewriter) const override {
    std::string ptxAsm = op.getRelaxed()
                             ? "barrier.cluster.arrive.relaxed.aligned;"
                             : "barrier.cluster.arrive.aligned;";
    return rewriteAsPtxAsm(op, rewriter, std::move(ptxAsm));
  }
};

class StoreMatrixOpPattern : public OpRewritePattern<ttn::StoreMatrixOp> {
public:
  using OpRewritePattern<ttn::StoreMatrixOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::StoreMatrixOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op));
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::StoreMatrixOp op) const {
    OperandsAndConstraints operandsAndTypes;
    auto addr = op.getAddr();
    auto datas = op.getDatas();
    operandsAndTypes.push_back({addr, "r"});
    for (unsigned i = 0; i < datas.size(); i++) {
      operandsAndTypes.push_back({datas[i], "r"});
    }
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::StoreMatrixOp op) const {
    auto datas = op.getDatas();
    std::string ptxAsm;
    switch (datas.size()) {
    case 1:
      ptxAsm = "stmatrix.sync.aligned.m8n8.x1.shared.b16 [$0], {$1};";
      break;
    case 2:
      ptxAsm = "stmatrix.sync.aligned.m8n8.x2.shared.b16 [$0], {$1, $2};";
      break;
    case 4:
      ptxAsm =
          "stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1, $2, $3, $4};";
      break;
    default:
      assert(false && "Invalid size");
    }
    return ptxAsm;
  }
};

class WGMMAWaitGroupOpPattern : public OpRewritePattern<ttn::WGMMAWaitGroupOp> {
public:
  using OpRewritePattern<ttn::WGMMAWaitGroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::WGMMAWaitGroupOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op),
                           getOutputConstraints(op));
  }

  Constraints getOutputConstraints(ttn::WGMMAWaitGroupOp op) const {
    auto outputStructType = cast<LLVM::LLVMStructType>(op.getType());
    uint32_t numOutputRegs = outputStructType.getBody().size();
    std::string output =
        outputStructType.getBody().front().isF32() ? "=f" : "=r";
    return Constraints(numOutputRegs, output);
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::WGMMAWaitGroupOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto input = op.getInput();
    operandsAndConstraints.push_back({input, "0"});
    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMAWaitGroupOp op) const {
    auto outputStructType = dyn_cast<LLVM::LLVMStructType>(op.getType());
    uint32_t numCRegs = outputStructType.getBody().size();
    std::string args = "";
    uint32_t asmOpIdx = 0;
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    auto ptxAsm = "// wait for regs: " + args + "\n\t" +
                  "wgmma.wait_group.sync.aligned #pendings;";
    return ptxAsm;
  }
};

class WGMMAOpPattern : public OpRewritePattern<ttn::WGMMAOp> {
public:
  using OpRewritePattern<ttn::WGMMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::WGMMAOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op),
                           getOutputConstraints(op));
  }

  std::vector<std::string> getOutputConstraints(ttn::WGMMAOp op) const {
    // TODO (zahi): Return type must always be a struct for wgmma, currently
    // we rely on the size of output constraints vector to determine whether
    // the output is a struct or not. We should find a way to pass this info
    auto resultType = op.getType();

    auto outputStructType = dyn_cast<LLVM::LLVMStructType>(resultType);
    uint32_t numOutputRegs = outputStructType.getBody().size();
    std::string output =
        outputStructType.getBody().front().isF32() ? "=f" : "=r";
    return std::vector<std::string>(numOutputRegs, output);
  }

  OperandsAndConstraints getOperandsAndConstraints(ttn::WGMMAOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto opC = op.getOpC();
    auto typeA = opA.getType();

    auto structTypeA = dyn_cast<LLVM::LLVMStructType>(typeA);

    // TODO (zahi): is this the best way to tie inputs/outputs ?
    if (opC)
      operandsAndConstraints.push_back({opC, "0"});

    if (structTypeA) {
      operandsAndConstraints.push_back({opA, "r"});
    } else {
      operandsAndConstraints.push_back({opA, "l"});
    }

    // Operand B (must be `desc`)
    operandsAndConstraints.push_back({opB, "l"});
    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMAOp op) const {
    using namespace ttn;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto m = op.getM();
    auto n = op.getN();
    auto k = op.getK();
    auto eltTypeC = op.getEltTypeC();
    auto eltTypeA = op.getEltTypeA();
    auto eltTypeB = op.getEltTypeB();
    auto layoutA = op.getLayoutA();
    auto layoutB = op.getLayoutB();

    // Register checks
    auto typeA = opA.getType();
    auto typeB = opB.getType();
    auto typeOutput = op.getType();
    auto structTypeA = dyn_cast<LLVM::LLVMStructType>(typeA);
    auto structTypeB = dyn_cast<LLVM::LLVMStructType>(typeB);
    auto structTypeOutput = dyn_cast<LLVM::LLVMStructType>(typeOutput);
    assert(!structTypeB && "Operand B can not be registers");
    assert(structTypeOutput && "Output and C operand must be registers");

    // Element type, MNK shape and transposing support check
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
    bool transA = layoutA == WGMMALayout::col;
    bool transB = layoutB == WGMMALayout::row;
    bool supported = false, needTransArgs = false, floatTypeWGMMA = false;
    assert(m % 8 == 0 && n % 8 == 0 && k % 8 == 0);
    // Below instructions do support transposing, must pass `trans` arguments
    supported |=
        (eltTypeA == WGMMAEltType::f16) && (eltTypeB == WGMMAEltType::f16) &&
        (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
        (m == 64 && 8 <= n && n <= 256 && k == 16);
    supported |= (eltTypeA == WGMMAEltType::bf16) &&
                 (eltTypeB == WGMMAEltType::bf16) &&
                 (eltTypeC == WGMMAEltType::f32) &&
                 (m == 64 && 8 <= n && n <= 256 && k == 16);
    needTransArgs = supported;
    floatTypeWGMMA = supported;
    // Below instructions do not support transposing
    if (!supported && !transA && !transB) {
      supported |= (eltTypeA == WGMMAEltType::tf32) &&
                   (eltTypeB == WGMMAEltType::tf32) &&
                   (eltTypeC == WGMMAEltType::f32) &&
                   (m == 64 && 8 <= n && n <= 256 && k == 8);
      supported |=
          (eltTypeA == WGMMAEltType::e4m3 || eltTypeA == WGMMAEltType::e5m2) &&
          (eltTypeB == WGMMAEltType::e4m3 || eltTypeB == WGMMAEltType::e5m2) &&
          (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
          (m == 64 && 8 <= n && n <= 256 && k == 32);
      floatTypeWGMMA = supported;
      // Below instructions are integer-based
      supported |= (eltTypeA == WGMMAEltType::s8) &&
                   (eltTypeB == WGMMAEltType::s8) &&
                   (eltTypeC == WGMMAEltType::s32) &&
                   (m == 64 && 8 <= n && n <= 224 && k == 32);
    }
    assert(supported && "WGMMA type or shape is not supported");

    // Operands
    uint32_t asmOpIdx = 0;
    std::string args = "";

    // Output and operand C
    uint32_t numCRegs = structTypeOutput.getBody().size();

    args += "{";
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    args += "}, ";

    if (op.getOpC())
      asmOpIdx += numCRegs;

    // Operand A
    if (structTypeA) {
      uint32_t numARegs = structTypeA.getBody().size();
      args += "{";
      for (uint32_t i = 0; i < numARegs; ++i) {
        args +=
            "$" + std::to_string(asmOpIdx++) + (i == numARegs - 1 ? "" : ",");
      }
      args += "}, ";
    } else {
      args += "$" + std::to_string(asmOpIdx++) + ", ";
    }

    // Operand B (must be `desc`)
    args += "$" + std::to_string(asmOpIdx++) + ", ";

    // `scale-d` is 1 if we have a C operand.
    args += op.getOpC() ? "1" : "0";

    // `imm-scale-a`, and `imm-scale-b` are 1 by default only for float-based
    // WGMMA
    if (floatTypeWGMMA)
      args += ", 1, 1";

    // Push `trans-a` and `trans-b` args if needed (determined as constant)
    if (needTransArgs) {
      if (!structTypeA)
        args += ", " + std::to_string(transA);
      args += ", " + std::to_string(transB);
    }

    auto ptxAsm = "wgmma.mma_async.sync.aligned"
                  ".m" +
                  std::to_string(m) + "n" + std::to_string(n) + "k" +
                  std::to_string(k) + "." + stringifyEnum(eltTypeC).str() +
                  "." + stringifyEnum(eltTypeA).str() + "." +
                  stringifyEnum(eltTypeB).str() + " " + args + ";";
    return ptxAsm;
  }
};

class ConvertNVGPUToLLVM : public ConvertNVGPUToLLVMBase<ConvertNVGPUToLLVM> {

public:
  explicit ConvertNVGPUToLLVM() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

#define POPULATE_NVGPU_OP(SRC_OP, ASM)                                         \
  patterns.add<NVGPUOpGenericPattern<SRC_OP>>(context, ASM, Constraints(),     \
                                              Constraints());
    POPULATE_NVGPU_OP(ttn::WGMMAFenceOp, Wgmma_Fence_Op)
    POPULATE_NVGPU_OP(ttn::WGMMACommitGroupOp, Wgmma_Commit_Group_Op)
    POPULATE_NVGPU_OP(ttn::ClusterWaitOp, Cluster_Wait_Op)
#undef POPULATE_NVGPU_OP
    patterns.add<NVGPUOpGenericPattern<ttn::ClusterCTAIdOp>>(
        context, Cluster_Cta_Id_Op, Constraints({"=r"}), Constraints());

    patterns
        .add<FenceAsyncSharedOpPattern, StoreMatrixOpPattern,
             ClusterArriveOpPattern, WGMMAOpPattern, WGMMAWaitGroupOpPattern>(
            context);

    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

LogicalResult
nvgpu::rewriteAsPtxAsm(Operation *op, PatternRewriter &rewriter,
                       std::string ptxAsm,
                       const OperandsAndConstraints &operandsAndConstraints,
                       const Constraints &outputConstraints) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  ptxAsm = patchPtxAsm(op, std::move(ptxAsm));
  auto hasSideEffects = !isMemoryEffectFree(op);

  PTXBuilder ptxBuilder;
  auto ptxOutputs = getPtxOutputs(outputConstraints, ptxBuilder);
  auto ptxOperands =
      getPtxOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
  SmallVector<PTXBuilder::Operand *> outputsAndOperands = ptxOutputs;
  outputsAndOperands.append(ptxOperands.begin(), ptxOperands.end());
  auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
  ptxInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
  auto retTy =
      op->getNumResults() == 0 ? void_ty(ctx) : op->getResult(0).getType();
  auto res = ptxBuilder.launch(rewriter, loc, retTy,
                               /*hasSideEffects*/ hasSideEffects);
  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOp(op, res);
  }

  return success();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUToLLVMPass() {
  return std::make_unique<::ConvertNVGPUToLLVM>();
}

} // namespace triton
} // namespace mlir
