add_triton_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  Coalesce.cpp
  F32DotTC.cpp
  ReduceDataDuplication.cpp
  OptimizeDotOperands.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/OuterLoopPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Pipeliner/PipeliningUtility.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  Utility.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
