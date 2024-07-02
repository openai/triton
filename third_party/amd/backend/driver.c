#define __HIP_PLATFORM_AMD__
// clang-format off
// hip_depreated.h needs definitions from hip_runtime.h.
#include <hip/hip_runtime.h>
#include <hip/hip_deprecated.h>
// clang-format on
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// The list of paths to search for the HIP runtime library. The caller Python
// code should substitute the search path placeholder.
static const char *hipLibSearchPaths[] = {"/*py_libhip_search_path*/"};

// The list of HIP dynamic library symbols and their signature we are interested
// in this file.
// |FOR_EACH_ERR_FN| is a macro to process APIs that return hipError_t;
// |FOR_EACH_STR_FN| is a macro to process APIs that return const char *.
//
// HIP 6.0 introduced an updated hipGetDeviceProperties API under a new symbol,
// hipGetDevicePropertiesR0600. However, the associated hipDeviceProp_t was
// directly updated with breaking changes to match hipGetDevicePropertiesR0600
// in the header file. We include the header file from HIP 6.0. So here if we
// use hipGetDeviceProperties together with hipDeviceProp_t we will use the
// old API with a new struct definition and mess up the interpretation.
//
// This is a known issue: https://github.com/ROCm/ROCm/issues/2728.
//
// For now explicitly defer to the old hipDeviceProp_t struct. This should work
// for both 5.x and 6.x. In the long term we need to switch to use
// hipGetProcAddress once available:
// https://github.com/ROCm/clr/commit/0479cdb3dd30ef58718cad44e424bd793c394cc0
#define HIP_SYMBOL_LIST(FOR_EACH_ERR_FN, FOR_EACH_STR_FN)                      \
  FOR_EACH_STR_FN(hipGetErrorString, hipError_t hipError)                      \
  FOR_EACH_ERR_FN(hipGetDeviceProperties, hipDeviceProp_tR0000 *prop,          \
                  int deviceId)                                                \
  FOR_EACH_ERR_FN(hipModuleLoadDataEx, hipModule_t *module, const void *image, \
                  unsigned int numOptions, hipJitOption *options,              \
                  void **optionValues)                                         \
  FOR_EACH_ERR_FN(hipModuleGetFunction, hipFunction_t *function,               \
                  hipModule_t module, const char *kname)                       \
  FOR_EACH_ERR_FN(hipFuncGetAttribute, int *, hipFunction_attribute attr,      \
                  hipFunction_t function)

// The HIP symbol table for holding resolved dynamic library symbols.
struct HIPSymbolTable {
#define DEFINE_EACH_ERR_FIELD(hipSymbolName, ...)                              \
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#define DEFINE_EACH_STR_FIELD(hipSymbolName, ...)                              \
  const char *(*hipSymbolName)(__VA_ARGS__);

  HIP_SYMBOL_LIST(DEFINE_EACH_ERR_FIELD, DEFINE_EACH_STR_FIELD)
};

static struct HIPSymbolTable hipSymbolTable;

bool initSymbolTable() {
  // Use the HIP runtime library loaded into the existing process if it exits.
  void *lib = dlopen("libamdhip64.so", RTLD_NOLOAD);
  if (lib) {
    // printf("[triton] chosen loaded libamdhip64.so in the process\n");
  }

  // Otherwise, go through the list of search paths to dlopen the first HIP
  // driver library.
  if (!lib) {
    int n = sizeof(hipLibSearchPaths) / sizeof(hipLibSearchPaths[0]);
    for (int i = 0; i < n; ++i) {
      void *handle = dlopen(hipLibSearchPaths[i], RTLD_LAZY | RTLD_LOCAL);
      if (handle) {
        lib = handle;
        // printf("[triton] chosen %s\n", hipLibSearchPaths[i]);
      }
    }
  }
  if (!lib) {
    PyErr_SetString(PyExc_RuntimeError, "cannot open libamdhip64.so");
    return false;
  }

  // Resolve all symbols we are interested in.
  dlerror(); // Clear existing errors
  const char *error = NULL;
#define QUERY_EACH_FN(hipSymbolName, ...)                                      \
  *(void **)&hipSymbolTable.hipSymbolName = dlsym(lib, #hipSymbolName);        \
  error = dlerror();                                                           \
  if (error) {                                                                 \
    PyErr_SetString(PyExc_RuntimeError,                                        \
                    "cannot query " #hipSymbolName " from libamdhip64.so");    \
    dlclose(lib);                                                              \
    return false;                                                              \
  }

  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)

  return true;
}

static inline void gpuAssert(hipError_t code, const char *file, int line) {
  {
    if (code != HIP_SUCCESS) {
      {
        const char *prefix = "Triton Error [HIP]: ";
        const char *str = hipSymbolTable.hipGetErrorString(code);
        char err[1024] = {0};
        snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str);
        PyGILState_STATE gil_state;
        gil_state = PyGILState_Ensure();
        PyErr_SetString(PyExc_RuntimeError, err);
        PyGILState_Release(gil_state);
      }
    }
  }
}

#define HIP_CHECK(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
    if (PyErr_Occurred())                                                      \
      return NULL;                                                             \
  }

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  hipDeviceProp_tR0000 props;
  HIP_CHECK(hipSymbolTable.hipGetDeviceProperties(&props, device_id));

  // create a struct to hold device properties
  return Py_BuildValue(
      "{s:i, s:i, s:i, s:i, s:i, s:i, s:s, s:i, s:i}", "max_shared_mem",
      props.sharedMemPerBlock, "max_num_regs", props.regsPerBlock,
      "multiprocessor_count", props.multiProcessorCount, "sm_clock_rate",
      props.clockRate, "mem_clock_rate", props.memoryClockRate, "mem_bus_width",
      props.memoryBusWidth, "arch", props.gcnArchName, "warpSize",
      props.warpSize, "max_threads_per_sm", props.maxThreadsPerMultiProcessor);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }

  // set HIP options
  hipJitOption opt[] = {hipJitOptionErrorLogBufferSizeBytes,
                        hipJitOptionErrorLogBuffer,
                        hipJitOptionInfoLogBufferSizeBytes,
                        hipJitOptionInfoLogBuffer, hipJitOptionLogVerbose};
  const unsigned int errbufsize = 8192;
  const unsigned int logbufsize = 8192;
  char _err[errbufsize];
  char _log[logbufsize];
  void *optval[] = {(void *)(uintptr_t)errbufsize, (void *)_err,
                    (void *)(uintptr_t)logbufsize, (void *)_log, (void *)1};

  // launch HIP Binary
  hipModule_t mod;
  hipFunction_t fun;
  HIP_CHECK(hipSymbolTable.hipModuleLoadDataEx(&mod, data, 5, opt, optval))
  HIP_CHECK(hipSymbolTable.hipModuleGetFunction(&fun, mod, name));

  // get allocated registers and spilled registers from the function
  int n_regs = 0;
  int n_spills = 0;
  hipSymbolTable.hipFuncGetAttribute(&n_regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, fun);
  hipSymbolTable.hipFuncGetAttribute(&n_spills,
                                     HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun);
  n_spills /= 4;
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided hsaco into HIP driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "hip_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_hip_utils(void) {
  if (!initSymbolTable()) {
    return NULL;
  }

  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
