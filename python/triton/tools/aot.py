import triton
import argparse
import triton._C.libtriton.triton as libtriton

if __name__ == '__main__':

  # valid source and target formats
  VALID_FORMATS = ['llvm-ir', 'ptx', 'triton-ir', 'triton-gpu-ir']

  # set up the argument parser
  # TODO: conditional requirements
  parser = argparse.ArgumentParser()
  parser.add_argument('src', help="Source file to compile")
  parser.add_argument('--target', required=True,
                      help="Target format, one of: " + ', '.join(VALID_FORMATS))
  parser.add_argument('-cc', '--compute-capability', type=int, required=True,
                      help="Compute capability to compile for")
  parser.add_argument('-ptx', '--ptx-version', type=int, required=True,
                      help="PTX version to compile for")

  # parse the args
  args = parser.parse_args()

  # TODO: clean-up and re-use triton.compiler primitive functions
  # check for validity of format arguments
  if args.target not in VALID_FORMATS:
    print("Invalid target format: " + args.target)
    exit(0)

  # parse source file to MLIR module
  context = libtriton.ir.context()
  module = libtriton.ir.parse_mlir_module(args.src, context)
  module.context = context

  # optimizer triton-ir
  module = triton.compiler.optimize_triton_ir(module)
  if args.target == 'triton-ir':
    print(module.str())
    exit(0)
  
  # triton-ir -> triton-gpu-ir
  module = triton.compiler.make_tritongpu_ir(module, num_warps=4)
  module = triton.compiler.optimize_tritongpu_ir(module, num_stages=3)
  if args.target == 'triton-gpu-ir':
    print(module.str())
    exit(0)
  
  # triton-gpu-ir -> llvm-ir
  module = triton.compiler.make_llvm_ir(module)
  if args.target == 'llvm-ir':
    print(module)
    exit(0)

  # llvm-ir -> ptx
  module = triton.compiler.make_ptx(module, compute_capability=args.compute_capability, ptx_version=args.ptx_version)
  assert args.target == 'ptx'
  print(module)
    



