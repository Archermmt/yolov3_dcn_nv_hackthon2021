
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import ctypes
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ctypes.CDLL('/usr/local/quake/lib/libops_trt.so')

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()

#[From Nv-example] Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine,stream):
  inputs = []
  outputs = []
  bindings = []
  for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))
  return inputs, outputs, bindings

#[From Nv-example] This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
  # Transfer input data to the GPU.
  [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
  # Run inference.
  context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
  # Transfer predictions back from the GPU.
  [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
  # Synchronize the stream
  stream.synchronize()
  # Return only the host outputs.
  return [out.host for out in outputs]

def init_trt_engine(trt_path):
  stream = cuda.Stream()
  engines={'test_1':{}}
  with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine=runtime.deserialize_cuda_engine(f.read())
  engines['test_1']['context']=engine.create_execution_context()
  engines['test_1']['inputs'],engines['test_1']['outputs'],engines['test_1']['bindings']=allocate_buffers(engine,stream)
  return engines,stream

#define the graph
def trt_infer(stream,engines,res_0,res_1,res_2):
  res_3=res_1.astype("float32")
  #copy inputs
  np.copyto(engines['test_1']['inputs'][0].host,res_2.flatten())
  np.copyto(engines['test_1']['inputs'][1].host,res_3.flatten())
  res_4=do_inference(engines['test_1']['context'],bindings=engines['test_1']['bindings'],inputs=engines['test_1']['inputs'],outputs=engines['test_1']['outputs'],stream=stream,batch_size=res_3.shape[0])
  res_4[0]=res_4[0].reshape((-1,80,100,5))
  res_4[1]=res_4[1].reshape((-1,80))
  res_5=res_4[1].astype("int32")
  return res_4[0],res_5,res_0

