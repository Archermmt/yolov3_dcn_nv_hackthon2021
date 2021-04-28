import numpy as np
import os
import sys
import time
import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import argparse
parser=argparse.ArgumentParser(description='tensorrt runtime test')
parser.add_argument('--repeat_num',default=1000,type=int,help='Repeat number for testing')
parser.add_argument('--batch_size',default=1,type=int,help='Batch size for testing')
parser.add_argument('--engine',default="../../network/engine_code/yolov3_dcn_int8/yolov3_dcn_1.trt",type=str,help='engine file path')
ctypes.CDLL('/usr/local/quake/lib/libops_trt.so')
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
args,unknown=parser.parse_known_args()

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

#define the graph
def test(stream,engines,res_0,res_1,res_2):
  res_3=res_1.astype("float32")
  #copy inputs
  np.copyto(engines['test_1']['inputs'][0].host,res_2.flatten())
  np.copyto(engines['test_1']['inputs'][1].host,res_3.flatten())
  res_4=do_inference(engines['test_1']['context'],bindings=engines['test_1']['bindings'],inputs=engines['test_1']['inputs'],outputs=engines['test_1']['outputs'],stream=stream,batch_size=args.batch_size)
  res_4[0]=res_4[0].reshape((-1,80,100,5))
  res_4[1]=res_4[1].reshape((-1,80))
  res_5=res_4[1].astype("int32")
  return res_4[0],res_5,res_0

if __name__=='__main__':
  ##############################    load reference datas    ##############################
  datas={
    'im_id':np.fromfile(os.path.join('/usr/local/quake/datas/inference_datas/yolov3_dcn','im_id.bin'),dtype='int64').reshape([1, 1]),
    'im_size':np.fromfile(os.path.join('/usr/local/quake/datas/inference_datas/yolov3_dcn','im_size.bin'),dtype='int32').reshape([1, 2]),
    'image':np.fromfile(os.path.join('/usr/local/quake/datas/inference_datas/yolov3_dcn','image.bin'),dtype='float32').reshape([1, 3, 608, 608]),
    'multinms':np.fromfile(os.path.join('/usr/local/quake/datas/inference_datas/yolov3_dcn','multinms.bin'),dtype='float32').reshape([1, 80, 100, 5]),
    'multiclass_nms_0_num_int32':np.fromfile(os.path.join('/usr/local/quake/datas/inference_datas/yolov3_dcn','multiclass_nms_0_num_int32.bin'),dtype='int32').reshape([1, 80]),
    'im_id':np.fromfile(os.path.join('/usr/local/quake/datas/inference_datas/yolov3_dcn','im_id.bin'),dtype='int64').reshape([1, 1])}
  #align reference datas
  if datas['im_id'].shape[0]!=args.batch_size:
    repeat_data=np.repeat(datas['im_id'],1+args.batch_size//datas['im_id'].shape[0],axis=0)
    datas['im_id']=repeat_data[:args.batch_size]
  if datas['im_size'].shape[0]!=args.batch_size:
    repeat_data=np.repeat(datas['im_size'],1+args.batch_size//datas['im_size'].shape[0],axis=0)
    datas['im_size']=repeat_data[:args.batch_size]
  if datas['image'].shape[0]!=args.batch_size:
    repeat_data=np.repeat(datas['image'],1+args.batch_size//datas['image'].shape[0],axis=0)
    datas['image']=repeat_data[:args.batch_size]
  if datas['multinms'].shape[0]!=args.batch_size:
    repeat_data=np.repeat(datas['multinms'],1+args.batch_size//datas['multinms'].shape[0],axis=0)
    datas['multinms']=repeat_data[:args.batch_size]
  if datas['multiclass_nms_0_num_int32'].shape[0]!=args.batch_size:
    repeat_data=np.repeat(datas['multiclass_nms_0_num_int32'],1+args.batch_size//datas['multiclass_nms_0_num_int32'].shape[0],axis=0)
    datas['multiclass_nms_0_num_int32']=repeat_data[:args.batch_size]
  if datas['im_id'].shape[0]!=args.batch_size:
    repeat_data=np.repeat(datas['im_id'],1+args.batch_size//datas['im_id'].shape[0],axis=0)
    datas['im_id']=repeat_data[:args.batch_size]
  ##############################    load reference datas    ##############################

  stream = cuda.Stream()
  engines={'test_1':{}}
  with open(args.engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine=runtime.deserialize_cuda_engine(f.read())
  engines['test_1']['context']=engine.create_execution_context()
  engines['test_1']['inputs'],engines['test_1']['outputs'],engines['test_1']['bindings']=allocate_buffers(engine,stream)

  start=time.time()
  for i in range(args.repeat_num):
    results=test(stream,engines,datas["im_id"],datas["im_size"],datas["image"])
  end=time.time()
  print("[RESULTS] QPS : {}, AVG_TIME: {} ms".format(args.repeat_num*args.batch_size/(end-start),(end-start)*1000/args.repeat_num))

  '''
  ##############################    check reference results    ##############################
  result_mismatch=False
  print("#"*30+"    <Start> check reference results    "+"#"*30)
  if not dlr_utils.array_compare(results[0],datas['multinms'],'res_multinms','golden_multinms',0.05):
    result_mismatch=True
    print("[FAIL] Result mismatch, please check!")
  if not dlr_utils.array_compare(results[1],datas['multiclass_nms_0_num_int32'],'res_multiclass_nms_0_num_int32','golden_multiclass_nms_0_num_int32',0.05):
    result_mismatch=True
    print("[FAIL] Result mismatch, please check!")
  if not dlr_utils.array_compare(results[2],datas['im_id'],'res_im_id','golden_im_id',0.05):
    result_mismatch=True
    print("[FAIL] Result mismatch, please check!")
  if result_mismatch:
    raise Exception('Result mismatch')
  print("[RESULTS] QPS : {}, AVG_TIME: {} ms".format(args.repeat_num*args.batch_size/(end-start),(end-start)*1000/args.repeat_num))
  print("#"*30+"    <End> check reference results    "+"#"*30)
  ##############################    check reference results    ##############################
  '''
