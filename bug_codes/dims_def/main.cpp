#include "text_img_general_3.h"
#include "util/file_utils.h"
#include "util/base.h"
#include <chrono>
#include <atomic>

using namespace quake::framework::ops_lib;
DLRLogger dlr_logger=DLRLogger();

std::atomic<long> totalSuccessNum(0L);
std::atomic<long> totalSuccessTimeUsed(0L);

bool test(std::shared_ptr<ICudaEngine> engine,int batch_size,bool show_detail)
{
  auto context=TRTUniquePtr<IExecutionContext>(engine->createExecutionContext());
  if (!context){
    dlr_logger.log(ILogger::Severity::kERROR,"Create execution context failed!");
    return false;
  }

  bool passed=true;
  int failed_cnt=0;
  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // Malloc and copy buffers
  void* cpu_buffers[31];
  void* gpu_buffers[31];

  // Malloc inputs buffers
  const int input_0 = engine->getBindingIndex("Node_6070_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_0],batch_size*128*sizeof(float)));
  cpu_buffers[input_0]=malloc(batch_size*128*sizeof(float));
  const int input_1 = engine->getBindingIndex("Node_6134_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_1],batch_size*128*sizeof(float)));
  cpu_buffers[input_1]=malloc(batch_size*128*sizeof(float));
  const int input_2 = engine->getBindingIndex("Node_6198_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_2],batch_size*128*sizeof(float)));
  cpu_buffers[input_2]=malloc(batch_size*128*sizeof(float));
  const int input_3 = engine->getBindingIndex("Node_6262_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_3],batch_size*128*sizeof(float)));
  cpu_buffers[input_3]=malloc(batch_size*128*sizeof(float));
  const int input_4 = engine->getBindingIndex("Node_6326_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_4],batch_size*128*sizeof(float)));
  cpu_buffers[input_4]=malloc(batch_size*128*sizeof(float));
  const int input_5 = engine->getBindingIndex("Node_6390_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_5],batch_size*128*sizeof(float)));
  cpu_buffers[input_5]=malloc(batch_size*128*sizeof(float));
  const int input_6 = engine->getBindingIndex("Node_6454_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_6],batch_size*128*sizeof(float)));
  cpu_buffers[input_6]=malloc(batch_size*128*sizeof(float));
  const int input_7 = engine->getBindingIndex("Node_6518_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_7],batch_size*128*sizeof(float)));
  cpu_buffers[input_7]=malloc(batch_size*128*sizeof(float));
  const int input_8 = engine->getBindingIndex("Node_6582_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_8],batch_size*128*sizeof(float)));
  cpu_buffers[input_8]=malloc(batch_size*128*sizeof(float));
  const int input_9 = engine->getBindingIndex("Node_6646_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_9],batch_size*128*sizeof(float)));
  cpu_buffers[input_9]=malloc(batch_size*128*sizeof(float));
  const int input_10 = engine->getBindingIndex("Node_6710_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_10],batch_size*128*sizeof(float)));
  cpu_buffers[input_10]=malloc(batch_size*128*sizeof(float));
  const int input_11 = engine->getBindingIndex("Node_6774_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_11],batch_size*128*sizeof(float)));
  cpu_buffers[input_11]=malloc(batch_size*128*sizeof(float));
  const int input_12 = engine->getBindingIndex("Node_6838_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_12],batch_size*128*sizeof(float)));
  cpu_buffers[input_12]=malloc(batch_size*128*sizeof(float));
  const int input_13 = engine->getBindingIndex("Node_6902_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_13],batch_size*128*sizeof(float)));
  cpu_buffers[input_13]=malloc(batch_size*128*sizeof(float));
  const int input_14 = engine->getBindingIndex("Node_6966_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_14],batch_size*128*sizeof(float)));
  cpu_buffers[input_14]=malloc(batch_size*128*sizeof(float));
  const int input_15 = engine->getBindingIndex("Node_7023_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_15],batch_size*128*sizeof(float)));
  cpu_buffers[input_15]=malloc(batch_size*128*sizeof(float));
  const int input_16 = engine->getBindingIndex("Node_7087_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_16],batch_size*128*sizeof(float)));
  cpu_buffers[input_16]=malloc(batch_size*128*sizeof(float));
  const int input_17 = engine->getBindingIndex("Node_7151_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_17],batch_size*128*sizeof(float)));
  cpu_buffers[input_17]=malloc(batch_size*128*sizeof(float));
  const int input_18 = engine->getBindingIndex("Node_7215_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_18],batch_size*128*sizeof(float)));
  cpu_buffers[input_18]=malloc(batch_size*128*sizeof(float));
  const int input_19 = engine->getBindingIndex("Node_7279_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_19],batch_size*128*sizeof(float)));
  cpu_buffers[input_19]=malloc(batch_size*128*sizeof(float));
  const int input_20 = engine->getBindingIndex("Node_7361_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_20],batch_size*640*sizeof(float)));
  cpu_buffers[input_20]=malloc(batch_size*640*sizeof(float));
  const int input_21 = engine->getBindingIndex("Node_7365_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_21],batch_size*640*sizeof(float)));
  cpu_buffers[input_21]=malloc(batch_size*640*sizeof(float));
  const int input_22 = engine->getBindingIndex("Node_7369_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_22],batch_size*640*sizeof(float)));
  cpu_buffers[input_22]=malloc(batch_size*640*sizeof(float));
  const int input_23 = engine->getBindingIndex("Node_7373_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_23],batch_size*640*sizeof(float)));
  cpu_buffers[input_23]=malloc(batch_size*640*sizeof(float));
  const int input_24 = engine->getBindingIndex("Node_7377_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_24],batch_size*640*sizeof(float)));
  cpu_buffers[input_24]=malloc(batch_size*640*sizeof(float));
  const int input_25 = engine->getBindingIndex("Node_7381_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_25],batch_size*640*sizeof(float)));
  cpu_buffers[input_25]=malloc(batch_size*640*sizeof(float));
  const int input_26 = engine->getBindingIndex("Node_7385_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_26],batch_size*640*sizeof(float)));
  cpu_buffers[input_26]=malloc(batch_size*640*sizeof(float));
  const int input_27 = engine->getBindingIndex("Node_7389_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_27],batch_size*640*sizeof(float)));
  cpu_buffers[input_27]=malloc(batch_size*640*sizeof(float));
  const int input_28 = engine->getBindingIndex("Node_7720_concat_1");
  CHECK(cudaMalloc(&gpu_buffers[input_28],batch_size*16*sizeof(float)));
  cpu_buffers[input_28]=malloc(batch_size*16*sizeof(float));

  // Malloc the output buffers
  const int output_0 = engine->getBindingIndex("score");
  CHECK(cudaMalloc(&gpu_buffers[output_0],batch_size*2*sizeof(float)));
  cpu_buffers[output_0]=malloc(batch_size*2*sizeof(float));
  const int output_1 = engine->getBindingIndex("Node_7767");
  CHECK(cudaMalloc(&gpu_buffers[output_1],batch_size*1*sizeof(float)));
  cpu_buffers[output_1]=malloc(batch_size*1*sizeof(float));

  // Stream obj for data feeding
  DLRBatchStream data_stream("/data/DLRouter/test/DLRGen/Testset",batch_size,-1,"text_img_general_3.tensor_info");
  while (data_stream.next()){
    void* data=data_stream.getBatch();
    int pos=0;
    // memcopy from host to device
    CHECK(cudaMemcpyAsync(gpu_buffers[input_0],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_1],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_2],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_3],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_4],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_5],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_6],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_7],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_8],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_9],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_10],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_11],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_12],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_13],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_14],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_15],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_16],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_17],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_18],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_19],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_20],data+pos,batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*640*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_21],data+pos,batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*640*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_22],data+pos,batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*640*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_23],data+pos,batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*640*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_24],data+pos,batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*640*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_25],data+pos,batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*640*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_26],data+pos,batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*640*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_27],data+pos,batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*640*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_28],data+pos,batch_size*16*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*16*sizeof(float);
    cudaStreamSynchronize(stream);

    context->enqueue(batch_size,gpu_buffers,stream,nullptr);

    // memcopy from device to host
    CHECK(cudaMemcpyAsync(cpu_buffers[output_0],gpu_buffers[output_0],batch_size*2*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_1],gpu_buffers[output_1],batch_size*1*sizeof(float),cudaMemcpyDeviceToHost,stream));
    // verify the outputs
    failed_cnt=verify_buffer("score_batch_"+std::to_string(data_stream.getBatchesRead()),(float*)(data+pos),(float*)cpu_buffers[output_0],batch_size*2,0.05*100,show_detail);
    if(failed_cnt>batch_size*2*0.05)
      passed=false;
    pos+=batch_size*2*sizeof(float);
    failed_cnt=verify_buffer("Node_7767_batch_"+std::to_string(data_stream.getBatchesRead()),(float*)(data+pos),(float*)cpu_buffers[output_1],batch_size*1,0.05*100,show_detail);
    if(failed_cnt>batch_size*1*0.05)
      passed=false;
    pos+=batch_size*1*sizeof(float);
    // check results
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" Batch "+std::to_string(data_stream.getBatchesRead()));
  }

  // Release stream and device buffers
  cudaStreamDestroy(stream);
  for (int i=0;i<31;i++){
    CHECK(cudaFree(gpu_buffers[i]));
    free(cpu_buffers[i]);
  }
  return passed;
}

bool infer(std::shared_ptr<ICudaEngine> engine,int batch_size,int repeat_num,bool show_detail,double& totalTime)
{
  auto context=TRTUniquePtr<IExecutionContext>(engine->createExecutionContext());
  if (!context){
    dlr_logger.log(ILogger::Severity::kERROR,"Create execution context failed!");
    return false;
  }

  bool passed=true;
  int failed_cnt=0;
  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // Malloc and copy buffers
  void* cpu_buffers[31];
  void* gpu_buffers[31];

  // Malloc inputs buffers
  const int input_0 = engine->getBindingIndex("Node_6070_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_0],batch_size*128*sizeof(float)));
  cpu_buffers[input_0]=malloc(batch_size*128*sizeof(float));
  const int input_1 = engine->getBindingIndex("Node_6134_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_1],batch_size*128*sizeof(float)));
  cpu_buffers[input_1]=malloc(batch_size*128*sizeof(float));
  const int input_2 = engine->getBindingIndex("Node_6198_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_2],batch_size*128*sizeof(float)));
  cpu_buffers[input_2]=malloc(batch_size*128*sizeof(float));
  const int input_3 = engine->getBindingIndex("Node_6262_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_3],batch_size*128*sizeof(float)));
  cpu_buffers[input_3]=malloc(batch_size*128*sizeof(float));
  const int input_4 = engine->getBindingIndex("Node_6326_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_4],batch_size*128*sizeof(float)));
  cpu_buffers[input_4]=malloc(batch_size*128*sizeof(float));
  const int input_5 = engine->getBindingIndex("Node_6390_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_5],batch_size*128*sizeof(float)));
  cpu_buffers[input_5]=malloc(batch_size*128*sizeof(float));
  const int input_6 = engine->getBindingIndex("Node_6454_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_6],batch_size*128*sizeof(float)));
  cpu_buffers[input_6]=malloc(batch_size*128*sizeof(float));
  const int input_7 = engine->getBindingIndex("Node_6518_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_7],batch_size*128*sizeof(float)));
  cpu_buffers[input_7]=malloc(batch_size*128*sizeof(float));
  const int input_8 = engine->getBindingIndex("Node_6582_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_8],batch_size*128*sizeof(float)));
  cpu_buffers[input_8]=malloc(batch_size*128*sizeof(float));
  const int input_9 = engine->getBindingIndex("Node_6646_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_9],batch_size*128*sizeof(float)));
  cpu_buffers[input_9]=malloc(batch_size*128*sizeof(float));
  const int input_10 = engine->getBindingIndex("Node_6710_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_10],batch_size*128*sizeof(float)));
  cpu_buffers[input_10]=malloc(batch_size*128*sizeof(float));
  const int input_11 = engine->getBindingIndex("Node_6774_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_11],batch_size*128*sizeof(float)));
  cpu_buffers[input_11]=malloc(batch_size*128*sizeof(float));
  const int input_12 = engine->getBindingIndex("Node_6838_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_12],batch_size*128*sizeof(float)));
  cpu_buffers[input_12]=malloc(batch_size*128*sizeof(float));
  const int input_13 = engine->getBindingIndex("Node_6902_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_13],batch_size*128*sizeof(float)));
  cpu_buffers[input_13]=malloc(batch_size*128*sizeof(float));
  const int input_14 = engine->getBindingIndex("Node_6966_diff");
  CHECK(cudaMalloc(&gpu_buffers[input_14],batch_size*128*sizeof(float)));
  cpu_buffers[input_14]=malloc(batch_size*128*sizeof(float));
  const int input_15 = engine->getBindingIndex("Node_7023_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_15],batch_size*128*sizeof(float)));
  cpu_buffers[input_15]=malloc(batch_size*128*sizeof(float));
  const int input_16 = engine->getBindingIndex("Node_7087_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_16],batch_size*128*sizeof(float)));
  cpu_buffers[input_16]=malloc(batch_size*128*sizeof(float));
  const int input_17 = engine->getBindingIndex("Node_7151_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_17],batch_size*128*sizeof(float)));
  cpu_buffers[input_17]=malloc(batch_size*128*sizeof(float));
  const int input_18 = engine->getBindingIndex("Node_7215_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_18],batch_size*128*sizeof(float)));
  cpu_buffers[input_18]=malloc(batch_size*128*sizeof(float));
  const int input_19 = engine->getBindingIndex("Node_7279_reshape");
  CHECK(cudaMalloc(&gpu_buffers[input_19],batch_size*128*sizeof(float)));
  cpu_buffers[input_19]=malloc(batch_size*128*sizeof(float));
  const int input_20 = engine->getBindingIndex("Node_7361_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_20],batch_size*640*sizeof(float)));
  cpu_buffers[input_20]=malloc(batch_size*640*sizeof(float));
  const int input_21 = engine->getBindingIndex("Node_7365_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_21],batch_size*640*sizeof(float)));
  cpu_buffers[input_21]=malloc(batch_size*640*sizeof(float));
  const int input_22 = engine->getBindingIndex("Node_7369_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_22],batch_size*640*sizeof(float)));
  cpu_buffers[input_22]=malloc(batch_size*640*sizeof(float));
  const int input_23 = engine->getBindingIndex("Node_7373_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_23],batch_size*640*sizeof(float)));
  cpu_buffers[input_23]=malloc(batch_size*640*sizeof(float));
  const int input_24 = engine->getBindingIndex("Node_7377_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_24],batch_size*640*sizeof(float)));
  cpu_buffers[input_24]=malloc(batch_size*640*sizeof(float));
  const int input_25 = engine->getBindingIndex("Node_7381_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_25],batch_size*640*sizeof(float)));
  cpu_buffers[input_25]=malloc(batch_size*640*sizeof(float));
  const int input_26 = engine->getBindingIndex("Node_7385_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_26],batch_size*640*sizeof(float)));
  cpu_buffers[input_26]=malloc(batch_size*640*sizeof(float));
  const int input_27 = engine->getBindingIndex("Node_7389_concat_2");
  CHECK(cudaMalloc(&gpu_buffers[input_27],batch_size*640*sizeof(float)));
  cpu_buffers[input_27]=malloc(batch_size*640*sizeof(float));
  const int input_28 = engine->getBindingIndex("Node_7720_concat_1");
  CHECK(cudaMalloc(&gpu_buffers[input_28],batch_size*16*sizeof(float)));
  cpu_buffers[input_28]=malloc(batch_size*16*sizeof(float));

  // Malloc the output buffers
  const int output_0 = engine->getBindingIndex("score");
  CHECK(cudaMalloc(&gpu_buffers[output_0],batch_size*2*sizeof(float)));
  cpu_buffers[output_0]=malloc(batch_size*2*sizeof(float));
  const int output_1 = engine->getBindingIndex("Node_7767");
  CHECK(cudaMalloc(&gpu_buffers[output_1],batch_size*1*sizeof(float)));
  cpu_buffers[output_1]=malloc(batch_size*1*sizeof(float));

  // Copy input data from file to CPU buffers
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6070_diff.bin",(float*)cpu_buffers[input_0],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6134_diff.bin",(float*)cpu_buffers[input_1],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6198_diff.bin",(float*)cpu_buffers[input_2],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6262_diff.bin",(float*)cpu_buffers[input_3],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6326_diff.bin",(float*)cpu_buffers[input_4],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6390_diff.bin",(float*)cpu_buffers[input_5],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6454_diff.bin",(float*)cpu_buffers[input_6],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6518_diff.bin",(float*)cpu_buffers[input_7],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6582_diff.bin",(float*)cpu_buffers[input_8],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6646_diff.bin",(float*)cpu_buffers[input_9],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6710_diff.bin",(float*)cpu_buffers[input_10],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6774_diff.bin",(float*)cpu_buffers[input_11],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6838_diff.bin",(float*)cpu_buffers[input_12],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6902_diff.bin",(float*)cpu_buffers[input_13],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_6966_diff.bin",(float*)cpu_buffers[input_14],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7023_reshape.bin",(float*)cpu_buffers[input_15],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7087_reshape.bin",(float*)cpu_buffers[input_16],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7151_reshape.bin",(float*)cpu_buffers[input_17],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7215_reshape.bin",(float*)cpu_buffers[input_18],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7279_reshape.bin",(float*)cpu_buffers[input_19],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7361_concat_2.bin",(float*)cpu_buffers[input_20],batch_size*640,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7365_concat_2.bin",(float*)cpu_buffers[input_21],batch_size*640,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7369_concat_2.bin",(float*)cpu_buffers[input_22],batch_size*640,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7373_concat_2.bin",(float*)cpu_buffers[input_23],batch_size*640,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7377_concat_2.bin",(float*)cpu_buffers[input_24],batch_size*640,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7381_concat_2.bin",(float*)cpu_buffers[input_25],batch_size*640,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7385_concat_2.bin",(float*)cpu_buffers[input_26],batch_size*640,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7389_concat_2.bin",(float*)cpu_buffers[input_27],batch_size*640,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/dims_def/Node_7720_concat_1.bin",(float*)cpu_buffers[input_28],batch_size*16,show_detail);

  // Enqueue the task and record time
  std::chrono::steady_clock::time_point test_begin=std::chrono::steady_clock::now();
  for(int i=0;i<repeat_num;i++){
    std::chrono::steady_clock::time_point begin=std::chrono::steady_clock::now();
    // memcopy from host to device
    CHECK(cudaMemcpyAsync(gpu_buffers[input_0],cpu_buffers[input_0],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_1],cpu_buffers[input_1],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_2],cpu_buffers[input_2],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_3],cpu_buffers[input_3],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_4],cpu_buffers[input_4],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_5],cpu_buffers[input_5],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_6],cpu_buffers[input_6],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_7],cpu_buffers[input_7],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_8],cpu_buffers[input_8],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_9],cpu_buffers[input_9],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_10],cpu_buffers[input_10],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_11],cpu_buffers[input_11],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_12],cpu_buffers[input_12],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_13],cpu_buffers[input_13],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_14],cpu_buffers[input_14],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_15],cpu_buffers[input_15],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_16],cpu_buffers[input_16],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_17],cpu_buffers[input_17],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_18],cpu_buffers[input_18],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_19],cpu_buffers[input_19],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_20],cpu_buffers[input_20],batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_21],cpu_buffers[input_21],batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_22],cpu_buffers[input_22],batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_23],cpu_buffers[input_23],batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_24],cpu_buffers[input_24],batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_25],cpu_buffers[input_25],batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_26],cpu_buffers[input_26],batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_27],cpu_buffers[input_27],batch_size*640*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_28],cpu_buffers[input_28],batch_size*16*sizeof(float),cudaMemcpyHostToDevice,stream));
    cudaStreamSynchronize(stream);

    context->enqueue(batch_size,gpu_buffers,stream,nullptr);

    // memcopy from device to host
    CHECK(cudaMemcpyAsync(cpu_buffers[output_0],gpu_buffers[output_0],batch_size*2*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_1],gpu_buffers[output_1],batch_size*1*sizeof(float),cudaMemcpyDeviceToHost,stream));
    cudaStreamSynchronize(stream);
    std::chrono::steady_clock::time_point end=std::chrono::steady_clock::now();
    totalSuccessNum+=batch_size;
    totalSuccessTimeUsed+=std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
  }
  std::chrono::steady_clock::time_point test_end=std::chrono::steady_clock::now();
  totalTime = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_begin).count();

  //Verify the outputs
  failed_cnt=FileUtils::verify_buffer_with_file("score","/usr/local/quake/datas/inference_datas/dims_def/score.bin",(float*)cpu_buffers[output_0],batch_size*2,0.05*100,show_detail);
  if(failed_cnt>batch_size*2*0.05)
    passed=false;

  failed_cnt=FileUtils::verify_buffer_with_file("Node_7767","/usr/local/quake/datas/inference_datas/dims_def/Node_7767.bin",(float*)cpu_buffers[output_1],batch_size*1,0.05*100,show_detail);
  if(failed_cnt>batch_size*1*0.05)
    passed=false;

  // Release stream and device buffers
  cudaStreamDestroy(stream);
  for (int i=0;i<31;i++){
    CHECK(cudaFree(gpu_buffers[i]));
    free(cpu_buffers[i]);
  }
  return passed;
}

int main(int argc, char** argv){
  dlr_logger.log(ILogger::Severity::kINFO,"Building and running a GPU inference engine for text_img_general_3");
  text_img_general_3 sample;

  bool show_detail=false;
  int batch_size=1;
  int repeat_num=1;

  if(!show_detail){
    batch_size=argc>1? atoi(argv[1]):8;
    repeat_num=argc>2? atoi(argv[2]):1000;
  }

  //define the engine
  std::shared_ptr<ICudaEngine> engine;

  if(!FileUtils::file_exist("text_img_general_3.trt")){
    auto builder=TRTUniquePtr<IBuilder>(createInferBuilder(dlr_logger));  
    if (!builder){
      dlr_logger.log(ILogger::Severity::kERROR,"Create builder failed!");
      return -1;
    }

    auto network=TRTUniquePtr<INetworkDefinition>(builder->createNetwork());  
    if (!network){
      dlr_logger.log(ILogger::Severity::kERROR,"Create network failed!");
      return -1;
    }

    auto config=TRTUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config){
      dlr_logger.log(ILogger::Severity::kERROR,"Create config failed!");
      return -1;
    }

    //create input tensors and output tensors
    ITensor* inputs[29];
    ITensor* outputs[2];

    inputs[0]=network->addInput("Node_6070_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[1]=network->addInput("Node_6134_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[2]=network->addInput("Node_6198_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[3]=network->addInput("Node_6262_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[4]=network->addInput("Node_6326_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[5]=network->addInput("Node_6390_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[6]=network->addInput("Node_6454_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[7]=network->addInput("Node_6518_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[8]=network->addInput("Node_6582_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[9]=network->addInput("Node_6646_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[10]=network->addInput("Node_6710_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[11]=network->addInput("Node_6774_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[12]=network->addInput("Node_6838_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[13]=network->addInput("Node_6902_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[14]=network->addInput("Node_6966_diff",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[15]=network->addInput("Node_7023_reshape",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[16]=network->addInput("Node_7087_reshape",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[17]=network->addInput("Node_7151_reshape",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[18]=network->addInput("Node_7215_reshape",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[19]=network->addInput("Node_7279_reshape",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[20]=network->addInput("Node_7361_concat_2",DataType::kFLOAT,DimsHW{20,32});
    inputs[21]=network->addInput("Node_7365_concat_2",DataType::kFLOAT,DimsHW{20,32});
    inputs[22]=network->addInput("Node_7369_concat_2",DataType::kFLOAT,DimsHW{20,32});
    inputs[23]=network->addInput("Node_7373_concat_2",DataType::kFLOAT,DimsHW{20,32});
    inputs[24]=network->addInput("Node_7377_concat_2",DataType::kFLOAT,DimsHW{20,32});
    inputs[25]=network->addInput("Node_7381_concat_2",DataType::kFLOAT,DimsHW{20,32});
    inputs[26]=network->addInput("Node_7385_concat_2",DataType::kFLOAT,DimsHW{20,32});
    inputs[27]=network->addInput("Node_7389_concat_2",DataType::kFLOAT,DimsHW{20,32});
    inputs[28]=network->addInput("Node_7720_concat_1",DataType::kFLOAT,DimsHW{2,8});

    //build the sample
    if (!sample.build(builder,network,config,inputs,outputs,batch_size,dlr_logger)){
      dlr_logger.log(ILogger::Severity::kERROR,"Failed to build the model!");
      return -1;
    }

    //mark output tensors
    network->markOutput(*outputs[0]);
    network->markOutput(*outputs[1]);

    //serialize engine to file and read from file
    bool serialized=serialize_engine_to_file("text_img_general_3.trt",builder,network,config,dlr_logger);
    if (!serialized){
      dlr_logger.log(ILogger::Severity::kERROR,"Serialize failed!");
      return -1;
    }
    print_center("Engine of type float32 serialized to text_img_general_3.trt");
  }

  //load the engine from file
  bool deserialized=deserialize_engine_from_file("text_img_general_3.trt",engine,dlr_logger);
  if(!deserialized){
    dlr_logger.log(ILogger::Severity::kERROR,"Deserialize failed!");
    return -1;
  }
  
  if (!engine){
    dlr_logger.log(ILogger::Severity::kERROR,"Create engine failed!");
    return -1;
  }

  //inference and test testset/QPS
  bool passed=true;

  if(!show_detail){
    print_center("<Start> Test text_img_general_3 testset");
    passed=test(engine,batch_size,show_detail);
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" <End> Test text_img_general_3 testset");
    std::cout<<std::endl;
  }
  if(passed){
    print_center("<Start> Test text_img_general_3 inference");
    double totalTime;
    passed=infer(engine,batch_size,repeat_num,show_detail,totalTime);
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" <End> Test text_img_general_3 inference");
    std::cout<<std::endl;
    //report QPS
    double avgTime = double(totalSuccessTimeUsed)/(repeat_num*1000);
    double QPS = totalSuccessNum*1000000/totalTime;
    std::cout<<"Batch size "<<batch_size<<", Repeat num "<<repeat_num<<" -> avgTime : "<<avgTime<<" ms, QPS : "<<QPS<<std::endl;
    std::cerr<<"[RESULTS] QPS : "<<QPS<<std::endl;
  }

  //clean up the tools
  sample.clean_up();
  return passed ? 0:-1;
}