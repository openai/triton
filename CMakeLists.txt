cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(triton)
include(CTest)

if(NOT WIN32)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()



# Options
option(TRITON_BUILD_TUTORIALS "Build C++ Triton tutorials" ON)
option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)
set(TRITON_CODEGEN_BACKENDS "" CACHE STRING "Enable different codegen backends")

# Ensure Python3 vars are set correctly
# used conditionally in this file and by lit tests

# Customized release build type with assertions: TritonRelBuildWithAsserts
set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT WIN32)
  find_library(TERMINFO_LIBRARY tinfo)
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17")

# Third-party
include_directories(${PYBIND11_INCLUDE_DIR})



# #########
# LLVM
# #########
if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# Utilities
function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )


  # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_triton_object)

set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
  set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
  add_triton_object(${name} ${ARGN})
endfunction()


# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party) # Tablegen'd files

# link_directories(${LLVM_LIBRARY_DIR})
add_subdirectory(include)
add_subdirectory(lib)

# find_package(PythonLibs REQUIRED)
set(TRITON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# TODO: Figure out which target is sufficient to fix errors; triton is
# apparently not enough. Currently set linking libstdc++fs for all targets
# to support some old version GCC compilers like 8.3.0.
if (NOT WIN32 AND NOT APPLE)
  link_libraries(stdc++fs)
endif()


# -----

if (DEFINED TRITON_PLUGIN_DIRS)
  foreach(PLUGIN_DIR ${TRITON_PLUGIN_DIRS})
    # Read the plugin name under dir/backend/name.conf
    cmake_path(APPEND PLUGIN_DIR "backend" "name.conf" OUTPUT_VARIABLE PLUGIN_NAME_PATH)
    file(READ ${PLUGIN_NAME_PATH} PLUGIN_NAME)
    string(STRIP ${PLUGIN_NAME} PLUGIN_NAME)

    list(APPEND TRITON_PLUGIN_NAMES ${PLUGIN_NAME})

    # Include the plugin as part of the build, placing the build output under
    # ${TRITON_BINARY_DIR}/third_party/${PLUGIN_NAME}
    cmake_path(APPEND TRITON_BINARY_DIR "third_party" ${PLUGIN_NAME} OUTPUT_VARIABLE PLUGIN_DIR_BUILD_OUTPUT)
    message(STATUS "Building plugin '${PLUGIN_NAME}' from ${PLUGIN_DIR} with output ${PLUGIN_DIR_BUILD_OUTPUT}")
    add_subdirectory(${PLUGIN_DIR} ${PLUGIN_DIR_BUILD_OUTPUT})
  endforeach()
endif()

foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
  add_subdirectory(third_party/${CODEGEN_BACKEND})
endforeach()


get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(triton_plugins GLOBAL PROPERTY TRITON_PLUGINS)
set(TRITON_LIBRARIES
        ${triton_libs}
        ${triton_plugins}

        # mlir
        MLIRAMDGPUDialect
        MLIRNVVMDialect
        MLIRNVVMToLLVMIRTranslation
        MLIRGPUToNVVMTransforms
        MLIRGPUToGPURuntimeTransforms
        MLIRGPUTransforms
        MLIRIR
        MLIRControlFlowToLLVM
        MLIRBytecodeWriter
        MLIRPass
        MLIRTransforms
        MLIRLLVMDialect
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRMathToLLVM
        MLIRROCDLToLLVMIRTranslation
        MLIRGPUDialect
        MLIRSCFToControlFlow
        MLIRIndexToLLVM
        MLIRGPUToROCDLTransforms

        # LLVM
        LLVMPasses
        LLVMNVPTXCodeGen
        # LLVMNVPTXAsmPrinter
        LLVMAMDGPUCodeGen
        LLVMAMDGPUAsmParser

)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  list(APPEND TRITON_LIBRARIES
          LLVMAArch64CodeGen
          LLVMAArch64AsmParser
  )
else()
  list(APPEND TRITON_LIBRARIES
          LLVMX86CodeGen
          LLVMX86AsmParser
  )
endif()

# Define triton library
string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_CODEGEN_BACKENDS})

if (DEFINED TRITON_PLUGIN_NAMES)
  string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_BACKENDS_TUPLE} ${TRITON_PLUGIN_NAMES})
endif()

message(STATUS "Triton backends tuple: ${TRITON_BACKENDS_TUPLE}")

set(TRITON_BACKENDS_TUPLE "(${TRITON_BACKENDS_TUPLE})")
add_compile_definitions(TRITON_BACKENDS_TUPLE=${TRITON_BACKENDS_TUPLE})
# ------
if(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})

  if(PYTHON_INCLUDE_DIRS)
    include_directories(${PYTHON_INCLUDE_DIRS})
  else()
    find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
    include_directories(${Python3_INCLUDE_DIRS})
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${Python3_LIBRARIES})
    add_link_options(${Python3_LINK_OPTIONS})
  endif()

  add_library(triton SHARED ${PYTHON_SRC_PATH}/main.cc
                  ${PYTHON_SRC_PATH}/ir.cc
                  ${PYTHON_SRC_PATH}/passes.cc
                  ${PYTHON_SRC_PATH}/interpreter.cc
                  ${PYTHON_SRC_PATH}/llvm.cc)

  # Link triton with its dependencies
  target_link_libraries(triton PUBLIC ${TRITON_LIBRARIES})
  if(WIN32)
    target_link_libraries(triton PRIVATE ${CMAKE_DL_LIBS})
  else()
    target_link_libraries(triton PRIVATE z)
  endif()
  target_link_options(triton PRIVATE ${LLVM_LDFLAGS})
endif()

if(TRITON_BUILD_PYTHON_MODULE AND NOT WIN32)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

  # Check if the platform is MacOS
  if(APPLE)
    set(PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
  endif()

  target_link_libraries(triton PRIVATE ${PYTHON_LDFLAGS})
endif()

if(NOT TRITON_BUILD_PYTHON_MODULE)
  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    add_subdirectory(third_party/${CODEGEN_BACKEND})
  endforeach()
endif()

add_subdirectory(bin)
add_subdirectory(test)
add_subdirectory(unittest)
