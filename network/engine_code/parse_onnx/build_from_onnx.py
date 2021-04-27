import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import argparse
parser=argparse.ArgumentParser(description='parse from onnx test')
parser.add_argument('--onnx_model',default="/usr/local/quake/datas/models/paddle_det.onnx",type=str,help='onnx mode path')
args,unknown=parser.parse_known_args()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
  return val * 1 << 30

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
  with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, 'rb') as model:
      if not parser.parse(model.read()):
        print ('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
          print (parser.get_error(error))
        return None
    return builder.build_cuda_engine(network)

if __name__=='__main__':
  engine=build_engine_onnx(args.onnx_model)
  print("engine : "+str(engine))
