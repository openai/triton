/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#ifndef TDL_TOOLS_SYS_GETENV_HPP
#define TDL_TOOLS_SYS_GETENV_HPP

#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <set>
#include <string>

namespace mlir::triton {

inline const std::set<std::string> ENV_VARS = {
    "AMDGCN_ENABLE_DUMP",
    "DISABLE_FAST_REDUCTION",
    "DISABLE_LLVM_OPT",
    "DISABLE_MMA_V3",
    "DISABLE_PTXAS_OPT",
    "LLVM_IR_ENABLE_DUMP",
    "MLIR_ENABLE_DUMP",
    "TRITON_DISABLE_LINE_INFO",
    "TRITON_DISABLE_RESHAPE_ENCODING_INFERENCE",
    "MLIR_ENABLE_DIAGNOSTICS",
    "TRITON_ENABLE_LLVM_DEBUG",
    "DISABLE_LOC_DUMP",
    "USE_TTGIR_LOC",
};

namespace tools {

inline std::string getenv(const char *name) {
  const char *cstr = std::getenv(name);
  if (!cstr)
    return "";
  std::string result(cstr);
  return result;
}

inline bool getBoolEnv(const std::string &env) {
  assert(ENV_VARS.find(env.c_str()) != ENV_VARS.end() &&
         "envvar is not recognized");
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}

} // namespace tools
} // namespace mlir::triton

#endif
