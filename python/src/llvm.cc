#include "mlir/IR/BuiltinOps.h" // mlir::ModuleOp
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

using namespace llvm;

std::string translateLLVMIRToASM(llvm::Module &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion, bool isObject) {
  using namespace mlir;
  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
  if (triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    auto optIt = options.find("print-after-all");
    if (optIt != options.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  // module->print(llvm::outs(), nullptr);

  // create machine
  module.setTargetTriple(triple);
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetOptions opt;
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive)};
  // set data layout
  module.setDataLayout(machine->createDataLayout());
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    for (llvm::Function &f : module.functions())
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::legacy::PassManager pass;
    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
    pass.run(module);
  }
  return result;
}

using ret = py::return_value_policy;

void init_triton_llvm(py::module &&m) {

  py::class_<llvm::Metadata, std::unique_ptr<llvm::Metadata, py::nodelete>>(
      m, "metadata");
  py::class_<llvm::MDString, std::unique_ptr<llvm::MDString, py::nodelete>,
             llvm::Metadata>(m, "string_metadata");
  /*
      .def(py::init([](LLVMContext &context, const std::string &str) {
        return llvm::MDString::get(context, str);
      }));
      */

  py::class_<llvm::LLVMContext>(m, "context", py::module_local())
      .def(py::init<>())
      .def("get_string_metadata",
           [](llvm::LLVMContext &context, const std::string &str) {
             return llvm::MDString::get(context, str);
           });

  py::class_<llvm::Module::FunctionListType>(m, "function_list")
      .def(
          "__iter__",
          [](llvm::Module::FunctionListType &s) {
            return py::make_iterator(s.begin(), s.end());
          },
          py::keep_alive<0, 1>());

  // Flag behavior. See
  // https://llvm.org/doxygen/classllvm_1_1Module.html#a0a5c55e12c97b80021330fe82b642293
  // for details.
  py::class_<llvm::Module::ModFlagBehavior>(m, "module_flag_behavior",
                                            py::module_local());
  m.attr("MODULE_FLAG_BEHAVIOR_ERROR") = (llvm::Module::Error);

  py::class_<llvm::Module>(m, "module", py::module_local())
      .def(
          "__str__",
          [](llvm::Module *self) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << *self;
            return os.str();
          },
          ret::take_ownership)
      .def(
          "get_functions",
          [](llvm::Module *mod) -> llvm::Module::FunctionListType & {
            return mod->getFunctionList();
          },
          ret::reference_internal)
      .def(
          "get_function",
          [](llvm::Module *mod, std::string &name) -> llvm::Function * {
            return mod->getFunction(name);
          },
          ret::reference_internal)
      .def("add_flag",
           [](llvm::Module *mod, llvm::Module::ModFlagBehavior behavior,
              std::string &key, uint32_t value) {
             return mod->addModuleFlag(behavior, key, value);
           })
      .def("add_flag",
           [](llvm::Module *mod, llvm::Module::ModFlagBehavior behavior,
              std::string &key, llvm::Metadata *metadata) {
             return mod->addModuleFlag(behavior, key, metadata);
           });

  py::class_<llvm::Function>(m, "function", py::module_local())
      .def("set_calling_conv", &llvm::Function::setCallingConv)
      .def("add_fn_attr", [](llvm::Function *fn, std::string &name,
                             std::string &val) { fn->addFnAttr(name, val); })
      .def("has_public_visibility",
           [](llvm::Function *fn) {
             return fn->getVisibility() == llvm::GlobalValue::DefaultVisibility;
           })
      .def("is_declaration", &llvm::Function::isDeclaration);

  // optimization levels
  py::class_<llvm::OptimizationLevel>(m, "optimization_level",
                                      py::module_local());
  m.attr("OPTIMIZE_O0") = (llvm::OptimizationLevel::O0);
  m.attr("OPTIMIZE_O1") = (llvm::OptimizationLevel::O1);
  m.attr("OPTIMIZE_O2") = (llvm::OptimizationLevel::O2);
  m.attr("OPTIMIZE_O3") = (llvm::OptimizationLevel::O3);
  m.attr("OPTIMIZE_Os") = (llvm::OptimizationLevel::Os);
  m.attr("OPTIMIZE_Oz") = (llvm::OptimizationLevel::Oz);

  m.def(
      "to_module",
      [](mlir::ModuleOp &mod, llvm::LLVMContext &ctx) {
        return mlir::translateModuleToLLVMIR(mod, ctx);
      },
      py::keep_alive<0, 2>());

  m.def("optimize_module", [](llvm::Module *mod,
                              const llvm::OptimizationLevel &opt,
                              std::string triple, std::string proc,
                              std::string features,
                              const std::vector<std::string> &flags,
                              bool enable_fp_fusion) {
    if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
      return;

    for (llvm::Function &f : mod->functions())
      if (!f.hasFnAttribute(llvm::Attribute::NoInline))
        f.addFnAttr(llvm::Attribute::AlwaysInline);
    // verify and store llvm
    llvm::legacy::PassManager pm;
    pm.add(llvm::createAlwaysInlinerLegacyPass());
    pm.add(llvm::createVerifierPass());
    pm.run(*mod);

    using namespace llvm;
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PassInstrumentationCallbacks *instrCbPtr = nullptr;
    PassInstrumentationCallbacks passInstrCb;
    StandardInstrumentations standardInstr(mod->getContext(),
                                           /*DebugLogging*/ true);
    if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
      auto optMap = llvm::cl::getRegisteredOptions();
      auto optIt = optMap.find("print-after-all");
      if (optIt != optMap.end()) {
        auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
        *optPtr = true;
      }
      standardInstr.registerCallbacks(passInstrCb, &mam);
      instrCbPtr = &passInstrCb;
    }

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = true;
    // TODO: currently we run SLP vectorizer with an empty target machine. This
    // cause the vectorizer to create larger vector which could be bad.
    // Disabling it would currently cause regressions as this pass also applies
    // some scheduling that helps performance in some cases. We should work on
    // using NVPTX target instead and address the performance regressions with
    // some scheduling solution.
    tuningOptions.SLPVectorization = true;

    // create machine
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(triple, error);
    llvm::TargetOptions targetOptions;
    if (enable_fp_fusion)
      targetOptions.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    targetOptions.UnsafeFPMath = false;
    targetOptions.NoInfsFPMath = false;
    targetOptions.NoNaNsFPMath = true;
    targetOptions.TrapUnreachable = true;
    std::unique_ptr<llvm::TargetMachine> machine{
        target->createTargetMachine(triple, proc, features, {}, {})};
    // llvm::Reloc::PIC_, std::nullopt, llvm::CodeGenOptLevel::Aggressive)};
   
    mod->setTargetTriple(machine->getTargetTriple().getTriple());
    mod->setDataLayout(machine->createDataLayout());

    PassBuilder pb(machine.get(), tuningOptions, std::nullopt, instrCbPtr);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    /*
    pb.registerVectorizerStartEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
          // Triton generates large structure of scalars which may pessimise
          // optimizations, we run a pass to break up phi of struct to make sure
          // all the struct are removed for the following passes.
          fpm.addPass(BreakStructPhiNodesPass());
          fpm.addPass(InstCombinePass());
        });
        */
    mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
    mpm.run(*mod, mam);
  });

  m.def(
      "translate_to_asm",
      [](std::string llvmIR, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> py::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
          py::gil_scoped_release allow_threads;
          // create LLVM module from C++
          llvm::LLVMContext context;
          std::unique_ptr<llvm::MemoryBuffer> buffer =
              llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
          llvm::SMDiagnostic error;
          std::unique_ptr<llvm::Module> module =
              llvm::parseIR(buffer->getMemBufferRef(), error, context);
          if (!module) {
            llvm::report_fatal_error(
                "failed to parse IR: " + error.getMessage() +
                "lineno: " + std::to_string(error.getLineNo()));
          }
          obj = translateLLVMIRToASM(*module, triple, proc, features, flags,
                                     enable_fp_fusion, isObject);
        }
        if (isObject)
          return py::bytes(obj);
        else
          return py::str(obj);
      },
      ret::take_ownership);

  m.def("init_targets", []() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
    });
  });

  m.def("link_extern_lib", [](llvm::Module *mod, std::string path) {
    llvm::SMDiagnostic err;
    auto &ctx = mod->getContext();
    auto extMod = llvm::parseIRFile(path, err, ctx);
    if (!extMod) {
      llvm::errs() << "Failed to load " << path;
      return;
    }
    extMod->setTargetTriple(mod->getTargetTriple());
    extMod->setDataLayout(mod->getDataLayout());
    if (llvm::Linker::linkModules(
            *mod, std::move(extMod), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module &m, const StringSet<> &gvs) {
              llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
                return !gv.hasName() || (gvs.count(gv.getName()) == 0);
              });
            })) {
      llvm::errs() << "Failed to link " << path;
      return;
    }
  });
}
