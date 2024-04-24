add_triton_library(TritonAMDGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandHelper.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandWMMA.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM/WMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    GCNAsmFormat.cpp
    TritonGPUToLLVM.cpp
    BuiltinFuncToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    DecomposeUnsupportedConversions.cpp
    OptimizeLDSUsage.cpp
    OptimizeLDSUtility.cpp
    SPMDOpToLLVM.cpp

    DEPENDS
    TritonAMDGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
)
