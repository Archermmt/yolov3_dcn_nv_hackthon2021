#include "ssl_login_get_emb_1.h"
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
  void* cpu_buffers[38];
  void* gpu_buffers[38];

  // Malloc inputs buffers
  const int input_0 = engine->getBindingIndex("useragent_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_0],batch_size*128*sizeof(int)));
  cpu_buffers[input_0]=malloc(batch_size*128*sizeof(int));
  const int input_1 = engine->getBindingIndex("useragent_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_1],batch_size*128*sizeof(float)));
  cpu_buffers[input_1]=malloc(batch_size*128*sizeof(float));
  const int input_2 = engine->getBindingIndex("useragent_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_2],batch_size*128*sizeof(int)));
  cpu_buffers[input_2]=malloc(batch_size*128*sizeof(int));
  const int input_3 = engine->getBindingIndex("useragent_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_3],batch_size*128*sizeof(int)));
  cpu_buffers[input_3]=malloc(batch_size*128*sizeof(int));
  const int input_4 = engine->getBindingIndex("collina_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_4],batch_size*59*sizeof(int)));
  cpu_buffers[input_4]=malloc(batch_size*59*sizeof(int));
  const int input_5 = engine->getBindingIndex("collina_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_5],batch_size*59*sizeof(float)));
  cpu_buffers[input_5]=malloc(batch_size*59*sizeof(float));
  const int input_6 = engine->getBindingIndex("collina_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_6],batch_size*59*sizeof(int)));
  cpu_buffers[input_6]=malloc(batch_size*59*sizeof(int));
  const int input_7 = engine->getBindingIndex("collina_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_7],batch_size*59*sizeof(int)));
  cpu_buffers[input_7]=malloc(batch_size*59*sizeof(int));
  const int input_8 = engine->getBindingIndex("regsrc_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_8],batch_size*75*sizeof(int)));
  cpu_buffers[input_8]=malloc(batch_size*75*sizeof(int));
  const int input_9 = engine->getBindingIndex("regsrc_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_9],batch_size*75*sizeof(float)));
  cpu_buffers[input_9]=malloc(batch_size*75*sizeof(float));
  const int input_10 = engine->getBindingIndex("regsrc_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_10],batch_size*75*sizeof(int)));
  cpu_buffers[input_10]=malloc(batch_size*75*sizeof(int));
  const int input_11 = engine->getBindingIndex("regsrc_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_11],batch_size*75*sizeof(int)));
  cpu_buffers[input_11]=malloc(batch_size*75*sizeof(int));
  const int input_12 = engine->getBindingIndex("regip_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_12],batch_size*33*sizeof(float)));
  cpu_buffers[input_12]=malloc(batch_size*33*sizeof(float));
  const int input_13 = engine->getBindingIndex("Node_2592");
  CHECK(cudaMalloc(&gpu_buffers[input_13],batch_size*6336*sizeof(float)));
  cpu_buffers[input_13]=malloc(batch_size*6336*sizeof(float));
  const int input_14 = engine->getBindingIndex("Node_2607");
  CHECK(cudaMalloc(&gpu_buffers[input_14],batch_size*6336*sizeof(float)));
  cpu_buffers[input_14]=malloc(batch_size*6336*sizeof(float));
  const int input_15 = engine->getBindingIndex("Node_2622");
  CHECK(cudaMalloc(&gpu_buffers[input_15],batch_size*6336*sizeof(float)));
  cpu_buffers[input_15]=malloc(batch_size*6336*sizeof(float));
  const int input_16 = engine->getBindingIndex("loginip_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_16],batch_size*33*sizeof(float)));
  cpu_buffers[input_16]=malloc(batch_size*33*sizeof(float));
  const int input_17 = engine->getBindingIndex("Node_3436");
  CHECK(cudaMalloc(&gpu_buffers[input_17],batch_size*6336*sizeof(float)));
  cpu_buffers[input_17]=malloc(batch_size*6336*sizeof(float));
  const int input_18 = engine->getBindingIndex("Node_3451");
  CHECK(cudaMalloc(&gpu_buffers[input_18],batch_size*6336*sizeof(float)));
  cpu_buffers[input_18]=malloc(batch_size*6336*sizeof(float));
  const int input_19 = engine->getBindingIndex("Node_3466");
  CHECK(cudaMalloc(&gpu_buffers[input_19],batch_size*6336*sizeof(float)));
  cpu_buffers[input_19]=malloc(batch_size*6336*sizeof(float));
  const int input_20 = engine->getBindingIndex("mobile_origintoken_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_20],batch_size*12*sizeof(float)));
  cpu_buffers[input_20]=malloc(batch_size*12*sizeof(float));
  const int input_21 = engine->getBindingIndex("Node_4187");
  CHECK(cudaMalloc(&gpu_buffers[input_21],batch_size*2304*sizeof(float)));
  cpu_buffers[input_21]=malloc(batch_size*2304*sizeof(float));
  const int input_22 = engine->getBindingIndex("Node_4203");
  CHECK(cudaMalloc(&gpu_buffers[input_22],batch_size*2304*sizeof(float)));
  cpu_buffers[input_22]=malloc(batch_size*2304*sizeof(float));
  const int input_23 = engine->getBindingIndex("Node_4219");
  CHECK(cudaMalloc(&gpu_buffers[input_23],batch_size*2304*sizeof(float)));
  cpu_buffers[input_23]=malloc(batch_size*2304*sizeof(float));
  const int input_24 = engine->getBindingIndex("umid_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_24],batch_size*133*sizeof(int)));
  cpu_buffers[input_24]=malloc(batch_size*133*sizeof(int));
  const int input_25 = engine->getBindingIndex("umid_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_25],batch_size*133*sizeof(float)));
  cpu_buffers[input_25]=malloc(batch_size*133*sizeof(float));
  const int input_26 = engine->getBindingIndex("umid_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_26],batch_size*133*sizeof(int)));
  cpu_buffers[input_26]=malloc(batch_size*133*sizeof(int));
  const int input_27 = engine->getBindingIndex("umid_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_27],batch_size*133*sizeof(int)));
  cpu_buffers[input_27]=malloc(batch_size*133*sizeof(int));
  const int input_28 = engine->getBindingIndex("email_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_28],batch_size*35*sizeof(int)));
  cpu_buffers[input_28]=malloc(batch_size*35*sizeof(int));
  const int input_29 = engine->getBindingIndex("email_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_29],batch_size*35*sizeof(float)));
  cpu_buffers[input_29]=malloc(batch_size*35*sizeof(float));
  const int input_30 = engine->getBindingIndex("email_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_30],batch_size*35*sizeof(int)));
  cpu_buffers[input_30]=malloc(batch_size*35*sizeof(int));
  const int input_31 = engine->getBindingIndex("email_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_31],batch_size*35*sizeof(int)));
  cpu_buffers[input_31]=malloc(batch_size*35*sizeof(int));
  const int input_32 = engine->getBindingIndex("event_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_32],batch_size*16*sizeof(int)));
  cpu_buffers[input_32]=malloc(batch_size*16*sizeof(int));
  const int input_33 = engine->getBindingIndex("event_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_33],batch_size*16*sizeof(float)));
  cpu_buffers[input_33]=malloc(batch_size*16*sizeof(float));
  const int input_34 = engine->getBindingIndex("event_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_34],batch_size*16*sizeof(int)));
  cpu_buffers[input_34]=malloc(batch_size*16*sizeof(int));
  const int input_35 = engine->getBindingIndex("event_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_35],batch_size*16*sizeof(int)));
  cpu_buffers[input_35]=malloc(batch_size*16*sizeof(int));

  // Malloc the output buffers
  const int output_0 = engine->getBindingIndex("embedding");
  CHECK(cudaMalloc(&gpu_buffers[output_0],batch_size*192*sizeof(float)));
  cpu_buffers[output_0]=malloc(batch_size*192*sizeof(float));
  const int output_1 = engine->getBindingIndex("cls_layer");
  CHECK(cudaMalloc(&gpu_buffers[output_1],batch_size*768*sizeof(float)));
  cpu_buffers[output_1]=malloc(batch_size*768*sizeof(float));

  // Stream obj for data feeding
  DLRBatchStream data_stream("/data/DLRouter/test/DLRGen/Testset",batch_size,-1,"ssl_login_get_emb_1.tensor_info");
  while (data_stream.next()){
    void* data=data_stream.getBatch();
    int pos=0;
    // memcopy from host to device
    CHECK(cudaMemcpyAsync(gpu_buffers[input_0],data+pos,batch_size*128*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_1],data+pos,batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_2],data+pos,batch_size*128*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_3],data+pos,batch_size*128*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*128*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_4],data+pos,batch_size*59*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*59*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_5],data+pos,batch_size*59*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*59*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_6],data+pos,batch_size*59*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*59*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_7],data+pos,batch_size*59*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*59*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_8],data+pos,batch_size*75*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*75*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_9],data+pos,batch_size*75*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*75*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_10],data+pos,batch_size*75*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*75*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_11],data+pos,batch_size*75*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*75*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_12],data+pos,batch_size*33*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*33*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_13],data+pos,batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*6336*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_14],data+pos,batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*6336*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_15],data+pos,batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*6336*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_16],data+pos,batch_size*33*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*33*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_17],data+pos,batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*6336*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_18],data+pos,batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*6336*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_19],data+pos,batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*6336*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_20],data+pos,batch_size*12*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*12*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_21],data+pos,batch_size*2304*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*2304*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_22],data+pos,batch_size*2304*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*2304*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_23],data+pos,batch_size*2304*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*2304*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_24],data+pos,batch_size*133*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*133*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_25],data+pos,batch_size*133*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*133*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_26],data+pos,batch_size*133*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*133*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_27],data+pos,batch_size*133*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*133*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_28],data+pos,batch_size*35*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*35*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_29],data+pos,batch_size*35*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*35*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_30],data+pos,batch_size*35*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*35*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_31],data+pos,batch_size*35*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*35*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_32],data+pos,batch_size*16*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*16*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_33],data+pos,batch_size*16*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*16*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_34],data+pos,batch_size*16*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*16*sizeof(int);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_35],data+pos,batch_size*16*sizeof(int),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*16*sizeof(int);
    cudaStreamSynchronize(stream);

    context->enqueue(batch_size,gpu_buffers,stream,nullptr);

    // memcopy from device to host
    CHECK(cudaMemcpyAsync(cpu_buffers[output_0],gpu_buffers[output_0],batch_size*192*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_1],gpu_buffers[output_1],batch_size*768*sizeof(float),cudaMemcpyDeviceToHost,stream));
    // verify the outputs
    failed_cnt=verify_buffer("embedding_batch_"+std::to_string(data_stream.getBatchesRead()),(float*)(data+pos),(float*)cpu_buffers[output_0],batch_size*192,0.1*100,show_detail);
    if(failed_cnt>batch_size*192*0.1)
      passed=false;
    pos+=batch_size*192*sizeof(float);
    failed_cnt=verify_buffer("cls_layer_batch_"+std::to_string(data_stream.getBatchesRead()),(float*)(data+pos),(float*)cpu_buffers[output_1],batch_size*768,0.1*100,show_detail);
    if(failed_cnt>batch_size*768*0.1)
      passed=false;
    pos+=batch_size*768*sizeof(float);
    // check results
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" Batch "+std::to_string(data_stream.getBatchesRead()));
  }

  // Release stream and device buffers
  cudaStreamDestroy(stream);
  for (int i=0;i<38;i++){
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
  void* cpu_buffers[38];
  void* gpu_buffers[38];

  // Malloc inputs buffers
  const int input_0 = engine->getBindingIndex("useragent_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_0],batch_size*128*sizeof(int)));
  cpu_buffers[input_0]=malloc(batch_size*128*sizeof(int));
  const int input_1 = engine->getBindingIndex("useragent_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_1],batch_size*128*sizeof(float)));
  cpu_buffers[input_1]=malloc(batch_size*128*sizeof(float));
  const int input_2 = engine->getBindingIndex("useragent_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_2],batch_size*128*sizeof(int)));
  cpu_buffers[input_2]=malloc(batch_size*128*sizeof(int));
  const int input_3 = engine->getBindingIndex("useragent_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_3],batch_size*128*sizeof(int)));
  cpu_buffers[input_3]=malloc(batch_size*128*sizeof(int));
  const int input_4 = engine->getBindingIndex("collina_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_4],batch_size*59*sizeof(int)));
  cpu_buffers[input_4]=malloc(batch_size*59*sizeof(int));
  const int input_5 = engine->getBindingIndex("collina_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_5],batch_size*59*sizeof(float)));
  cpu_buffers[input_5]=malloc(batch_size*59*sizeof(float));
  const int input_6 = engine->getBindingIndex("collina_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_6],batch_size*59*sizeof(int)));
  cpu_buffers[input_6]=malloc(batch_size*59*sizeof(int));
  const int input_7 = engine->getBindingIndex("collina_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_7],batch_size*59*sizeof(int)));
  cpu_buffers[input_7]=malloc(batch_size*59*sizeof(int));
  const int input_8 = engine->getBindingIndex("regsrc_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_8],batch_size*75*sizeof(int)));
  cpu_buffers[input_8]=malloc(batch_size*75*sizeof(int));
  const int input_9 = engine->getBindingIndex("regsrc_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_9],batch_size*75*sizeof(float)));
  cpu_buffers[input_9]=malloc(batch_size*75*sizeof(float));
  const int input_10 = engine->getBindingIndex("regsrc_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_10],batch_size*75*sizeof(int)));
  cpu_buffers[input_10]=malloc(batch_size*75*sizeof(int));
  const int input_11 = engine->getBindingIndex("regsrc_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_11],batch_size*75*sizeof(int)));
  cpu_buffers[input_11]=malloc(batch_size*75*sizeof(int));
  const int input_12 = engine->getBindingIndex("regip_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_12],batch_size*33*sizeof(float)));
  cpu_buffers[input_12]=malloc(batch_size*33*sizeof(float));
  const int input_13 = engine->getBindingIndex("Node_2592");
  CHECK(cudaMalloc(&gpu_buffers[input_13],batch_size*6336*sizeof(float)));
  cpu_buffers[input_13]=malloc(batch_size*6336*sizeof(float));
  const int input_14 = engine->getBindingIndex("Node_2607");
  CHECK(cudaMalloc(&gpu_buffers[input_14],batch_size*6336*sizeof(float)));
  cpu_buffers[input_14]=malloc(batch_size*6336*sizeof(float));
  const int input_15 = engine->getBindingIndex("Node_2622");
  CHECK(cudaMalloc(&gpu_buffers[input_15],batch_size*6336*sizeof(float)));
  cpu_buffers[input_15]=malloc(batch_size*6336*sizeof(float));
  const int input_16 = engine->getBindingIndex("loginip_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_16],batch_size*33*sizeof(float)));
  cpu_buffers[input_16]=malloc(batch_size*33*sizeof(float));
  const int input_17 = engine->getBindingIndex("Node_3436");
  CHECK(cudaMalloc(&gpu_buffers[input_17],batch_size*6336*sizeof(float)));
  cpu_buffers[input_17]=malloc(batch_size*6336*sizeof(float));
  const int input_18 = engine->getBindingIndex("Node_3451");
  CHECK(cudaMalloc(&gpu_buffers[input_18],batch_size*6336*sizeof(float)));
  cpu_buffers[input_18]=malloc(batch_size*6336*sizeof(float));
  const int input_19 = engine->getBindingIndex("Node_3466");
  CHECK(cudaMalloc(&gpu_buffers[input_19],batch_size*6336*sizeof(float)));
  cpu_buffers[input_19]=malloc(batch_size*6336*sizeof(float));
  const int input_20 = engine->getBindingIndex("mobile_origintoken_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_20],batch_size*12*sizeof(float)));
  cpu_buffers[input_20]=malloc(batch_size*12*sizeof(float));
  const int input_21 = engine->getBindingIndex("Node_4187");
  CHECK(cudaMalloc(&gpu_buffers[input_21],batch_size*2304*sizeof(float)));
  cpu_buffers[input_21]=malloc(batch_size*2304*sizeof(float));
  const int input_22 = engine->getBindingIndex("Node_4203");
  CHECK(cudaMalloc(&gpu_buffers[input_22],batch_size*2304*sizeof(float)));
  cpu_buffers[input_22]=malloc(batch_size*2304*sizeof(float));
  const int input_23 = engine->getBindingIndex("Node_4219");
  CHECK(cudaMalloc(&gpu_buffers[input_23],batch_size*2304*sizeof(float)));
  cpu_buffers[input_23]=malloc(batch_size*2304*sizeof(float));
  const int input_24 = engine->getBindingIndex("umid_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_24],batch_size*133*sizeof(int)));
  cpu_buffers[input_24]=malloc(batch_size*133*sizeof(int));
  const int input_25 = engine->getBindingIndex("umid_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_25],batch_size*133*sizeof(float)));
  cpu_buffers[input_25]=malloc(batch_size*133*sizeof(float));
  const int input_26 = engine->getBindingIndex("umid_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_26],batch_size*133*sizeof(int)));
  cpu_buffers[input_26]=malloc(batch_size*133*sizeof(int));
  const int input_27 = engine->getBindingIndex("umid_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_27],batch_size*133*sizeof(int)));
  cpu_buffers[input_27]=malloc(batch_size*133*sizeof(int));
  const int input_28 = engine->getBindingIndex("email_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_28],batch_size*35*sizeof(int)));
  cpu_buffers[input_28]=malloc(batch_size*35*sizeof(int));
  const int input_29 = engine->getBindingIndex("email_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_29],batch_size*35*sizeof(float)));
  cpu_buffers[input_29]=malloc(batch_size*35*sizeof(float));
  const int input_30 = engine->getBindingIndex("email_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_30],batch_size*35*sizeof(int)));
  cpu_buffers[input_30]=malloc(batch_size*35*sizeof(int));
  const int input_31 = engine->getBindingIndex("email_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_31],batch_size*35*sizeof(int)));
  cpu_buffers[input_31]=malloc(batch_size*35*sizeof(int));
  const int input_32 = engine->getBindingIndex("event_origintoken_input_id_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_32],batch_size*16*sizeof(int)));
  cpu_buffers[input_32]=malloc(batch_size*16*sizeof(int));
  const int input_33 = engine->getBindingIndex("event_origintoken_input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_33],batch_size*16*sizeof(float)));
  cpu_buffers[input_33]=malloc(batch_size*16*sizeof(float));
  const int input_34 = engine->getBindingIndex("event_typetoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_34],batch_size*16*sizeof(int)));
  cpu_buffers[input_34]=malloc(batch_size*16*sizeof(int));
  const int input_35 = engine->getBindingIndex("event_relativepositiontoken_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_35],batch_size*16*sizeof(int)));
  cpu_buffers[input_35]=malloc(batch_size*16*sizeof(int));

  // Malloc the output buffers
  const int output_0 = engine->getBindingIndex("embedding");
  CHECK(cudaMalloc(&gpu_buffers[output_0],batch_size*192*sizeof(float)));
  cpu_buffers[output_0]=malloc(batch_size*192*sizeof(float));
  const int output_1 = engine->getBindingIndex("cls_layer");
  CHECK(cudaMalloc(&gpu_buffers[output_1],batch_size*768*sizeof(float)));
  cpu_buffers[output_1]=malloc(batch_size*768*sizeof(float));

  // Copy input data from file to CPU buffers
  FileUtils::read_file_to_buffer("Reference/useragent_origintoken_input_id_int32.bin",(int*)cpu_buffers[input_0],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("Reference/useragent_origintoken_input_mask_float32.bin",(float*)cpu_buffers[input_1],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("Reference/useragent_typetoken_int32.bin",(int*)cpu_buffers[input_2],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("Reference/useragent_relativepositiontoken_int32.bin",(int*)cpu_buffers[input_3],batch_size*128,show_detail);
  FileUtils::read_file_to_buffer("Reference/collina_origintoken_input_id_int32.bin",(int*)cpu_buffers[input_4],batch_size*59,show_detail);
  FileUtils::read_file_to_buffer("Reference/collina_origintoken_input_mask_float32.bin",(float*)cpu_buffers[input_5],batch_size*59,show_detail);
  FileUtils::read_file_to_buffer("Reference/collina_typetoken_int32.bin",(int*)cpu_buffers[input_6],batch_size*59,show_detail);
  FileUtils::read_file_to_buffer("Reference/collina_relativepositiontoken_int32.bin",(int*)cpu_buffers[input_7],batch_size*59,show_detail);
  FileUtils::read_file_to_buffer("Reference/regsrc_origintoken_input_id_int32.bin",(int*)cpu_buffers[input_8],batch_size*75,show_detail);
  FileUtils::read_file_to_buffer("Reference/regsrc_origintoken_input_mask_float32.bin",(float*)cpu_buffers[input_9],batch_size*75,show_detail);
  FileUtils::read_file_to_buffer("Reference/regsrc_typetoken_int32.bin",(int*)cpu_buffers[input_10],batch_size*75,show_detail);
  FileUtils::read_file_to_buffer("Reference/regsrc_relativepositiontoken_int32.bin",(int*)cpu_buffers[input_11],batch_size*75,show_detail);
  FileUtils::read_file_to_buffer("Reference/regip_origintoken_input_mask_float32.bin",(float*)cpu_buffers[input_12],batch_size*33,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_2592.bin",(float*)cpu_buffers[input_13],batch_size*6336,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_2607.bin",(float*)cpu_buffers[input_14],batch_size*6336,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_2622.bin",(float*)cpu_buffers[input_15],batch_size*6336,show_detail);
  FileUtils::read_file_to_buffer("Reference/loginip_origintoken_input_mask_float32.bin",(float*)cpu_buffers[input_16],batch_size*33,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_3436.bin",(float*)cpu_buffers[input_17],batch_size*6336,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_3451.bin",(float*)cpu_buffers[input_18],batch_size*6336,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_3466.bin",(float*)cpu_buffers[input_19],batch_size*6336,show_detail);
  FileUtils::read_file_to_buffer("Reference/mobile_origintoken_float32.bin",(float*)cpu_buffers[input_20],batch_size*12,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_4187.bin",(float*)cpu_buffers[input_21],batch_size*2304,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_4203.bin",(float*)cpu_buffers[input_22],batch_size*2304,show_detail);
  FileUtils::read_file_to_buffer("Reference/Node_4219.bin",(float*)cpu_buffers[input_23],batch_size*2304,show_detail);
  FileUtils::read_file_to_buffer("Reference/umid_origintoken_input_id_int32.bin",(int*)cpu_buffers[input_24],batch_size*133,show_detail);
  FileUtils::read_file_to_buffer("Reference/umid_origintoken_input_mask_float32.bin",(float*)cpu_buffers[input_25],batch_size*133,show_detail);
  FileUtils::read_file_to_buffer("Reference/umid_typetoken_int32.bin",(int*)cpu_buffers[input_26],batch_size*133,show_detail);
  FileUtils::read_file_to_buffer("Reference/umid_relativepositiontoken_int32.bin",(int*)cpu_buffers[input_27],batch_size*133,show_detail);
  FileUtils::read_file_to_buffer("Reference/email_origintoken_input_id_int32.bin",(int*)cpu_buffers[input_28],batch_size*35,show_detail);
  FileUtils::read_file_to_buffer("Reference/email_origintoken_input_mask_float32.bin",(float*)cpu_buffers[input_29],batch_size*35,show_detail);
  FileUtils::read_file_to_buffer("Reference/email_typetoken_int32.bin",(int*)cpu_buffers[input_30],batch_size*35,show_detail);
  FileUtils::read_file_to_buffer("Reference/email_relativepositiontoken_int32.bin",(int*)cpu_buffers[input_31],batch_size*35,show_detail);
  FileUtils::read_file_to_buffer("Reference/event_origintoken_input_id_int32.bin",(int*)cpu_buffers[input_32],batch_size*16,show_detail);
  FileUtils::read_file_to_buffer("Reference/event_origintoken_input_mask_float32.bin",(float*)cpu_buffers[input_33],batch_size*16,show_detail);
  FileUtils::read_file_to_buffer("Reference/event_typetoken_int32.bin",(int*)cpu_buffers[input_34],batch_size*16,show_detail);
  FileUtils::read_file_to_buffer("Reference/event_relativepositiontoken_int32.bin",(int*)cpu_buffers[input_35],batch_size*16,show_detail);

  // Enqueue the task and record time
  std::chrono::steady_clock::time_point test_begin=std::chrono::steady_clock::now();
  for(int i=0;i<repeat_num;i++){
    std::chrono::steady_clock::time_point begin=std::chrono::steady_clock::now();
    // memcopy from host to device
    CHECK(cudaMemcpyAsync(gpu_buffers[input_0],cpu_buffers[input_0],batch_size*128*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_1],cpu_buffers[input_1],batch_size*128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_2],cpu_buffers[input_2],batch_size*128*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_3],cpu_buffers[input_3],batch_size*128*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_4],cpu_buffers[input_4],batch_size*59*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_5],cpu_buffers[input_5],batch_size*59*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_6],cpu_buffers[input_6],batch_size*59*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_7],cpu_buffers[input_7],batch_size*59*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_8],cpu_buffers[input_8],batch_size*75*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_9],cpu_buffers[input_9],batch_size*75*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_10],cpu_buffers[input_10],batch_size*75*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_11],cpu_buffers[input_11],batch_size*75*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_12],cpu_buffers[input_12],batch_size*33*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_13],cpu_buffers[input_13],batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_14],cpu_buffers[input_14],batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_15],cpu_buffers[input_15],batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_16],cpu_buffers[input_16],batch_size*33*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_17],cpu_buffers[input_17],batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_18],cpu_buffers[input_18],batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_19],cpu_buffers[input_19],batch_size*6336*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_20],cpu_buffers[input_20],batch_size*12*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_21],cpu_buffers[input_21],batch_size*2304*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_22],cpu_buffers[input_22],batch_size*2304*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_23],cpu_buffers[input_23],batch_size*2304*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_24],cpu_buffers[input_24],batch_size*133*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_25],cpu_buffers[input_25],batch_size*133*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_26],cpu_buffers[input_26],batch_size*133*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_27],cpu_buffers[input_27],batch_size*133*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_28],cpu_buffers[input_28],batch_size*35*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_29],cpu_buffers[input_29],batch_size*35*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_30],cpu_buffers[input_30],batch_size*35*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_31],cpu_buffers[input_31],batch_size*35*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_32],cpu_buffers[input_32],batch_size*16*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_33],cpu_buffers[input_33],batch_size*16*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_34],cpu_buffers[input_34],batch_size*16*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_35],cpu_buffers[input_35],batch_size*16*sizeof(int),cudaMemcpyHostToDevice,stream));
    cudaStreamSynchronize(stream);

    context->enqueue(batch_size,gpu_buffers,stream,nullptr);

    // memcopy from device to host
    CHECK(cudaMemcpyAsync(cpu_buffers[output_0],gpu_buffers[output_0],batch_size*192*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_1],gpu_buffers[output_1],batch_size*768*sizeof(float),cudaMemcpyDeviceToHost,stream));
    cudaStreamSynchronize(stream);
    std::chrono::steady_clock::time_point end=std::chrono::steady_clock::now();
    totalSuccessNum+=batch_size;
    totalSuccessTimeUsed+=std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
  }
  std::chrono::steady_clock::time_point test_end=std::chrono::steady_clock::now();
  totalTime = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_begin).count();

  //Verify the outputs
  failed_cnt=FileUtils::verify_buffer_with_file("embedding","Reference/embedding.bin",(float*)cpu_buffers[output_0],batch_size*192,0.1*100,show_detail);
  if(failed_cnt>batch_size*192*0.1)
    passed=false;

  failed_cnt=FileUtils::verify_buffer_with_file("cls_layer","Reference/cls_layer.bin",(float*)cpu_buffers[output_1],batch_size*768,0.1*100,show_detail);
  if(failed_cnt>batch_size*768*0.1)
    passed=false;

  // Release stream and device buffers
  cudaStreamDestroy(stream);
  for (int i=0;i<38;i++){
    CHECK(cudaFree(gpu_buffers[i]));
    free(cpu_buffers[i]);
  }
  return passed;
}

int main(int argc, char** argv){
  dlr_logger.log(ILogger::Severity::kINFO,"Building and running a GPU inference engine for ssl_login_get_emb_1");
  ssl_login_get_emb_1 sample;

  bool show_detail=false;
  int batch_size=1;
  int repeat_num=1;

  if(!show_detail){
    batch_size=argc>1? atoi(argv[1]):8;
    repeat_num=argc>2? atoi(argv[2]):1000;
  }

  //define the engine
  std::shared_ptr<ICudaEngine> engine;

  if(!FileUtils::file_exist("ssl_login_get_emb_1.trt")){
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
    ITensor* inputs[36];
    ITensor* outputs[2];

    inputs[0]=network->addInput("useragent_origintoken_input_id_int32",DataType::kINT32,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[1]=network->addInput("useragent_origintoken_input_mask_float32",DataType::kFLOAT,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[2]=network->addInput("useragent_typetoken_int32",DataType::kINT32,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[3]=network->addInput("useragent_relativepositiontoken_int32",DataType::kINT32,Dims{1,{128},{DimensionType::kCHANNEL}});
    inputs[4]=network->addInput("collina_origintoken_input_id_int32",DataType::kINT32,Dims{1,{59},{DimensionType::kCHANNEL}});
    inputs[5]=network->addInput("collina_origintoken_input_mask_float32",DataType::kFLOAT,Dims{1,{59},{DimensionType::kCHANNEL}});
    inputs[6]=network->addInput("collina_typetoken_int32",DataType::kINT32,Dims{1,{59},{DimensionType::kCHANNEL}});
    inputs[7]=network->addInput("collina_relativepositiontoken_int32",DataType::kINT32,Dims{1,{59},{DimensionType::kCHANNEL}});
    inputs[8]=network->addInput("regsrc_origintoken_input_id_int32",DataType::kINT32,Dims{1,{75},{DimensionType::kCHANNEL}});
    inputs[9]=network->addInput("regsrc_origintoken_input_mask_float32",DataType::kFLOAT,Dims{1,{75},{DimensionType::kCHANNEL}});
    inputs[10]=network->addInput("regsrc_typetoken_int32",DataType::kINT32,Dims{1,{75},{DimensionType::kCHANNEL}});
    inputs[11]=network->addInput("regsrc_relativepositiontoken_int32",DataType::kINT32,Dims{1,{75},{DimensionType::kCHANNEL}});
    inputs[12]=network->addInput("regip_origintoken_input_mask_float32",DataType::kFLOAT,Dims{1,{33},{DimensionType::kCHANNEL}});
    inputs[13]=network->addInput("Node_2592",DataType::kFLOAT,DimsHW{33,192});
    inputs[14]=network->addInput("Node_2607",DataType::kFLOAT,DimsHW{33,192});
    inputs[15]=network->addInput("Node_2622",DataType::kFLOAT,DimsHW{33,192});
    inputs[16]=network->addInput("loginip_origintoken_input_mask_float32",DataType::kFLOAT,Dims{1,{33},{DimensionType::kCHANNEL}});
    inputs[17]=network->addInput("Node_3436",DataType::kFLOAT,DimsHW{33,192});
    inputs[18]=network->addInput("Node_3451",DataType::kFLOAT,DimsHW{33,192});
    inputs[19]=network->addInput("Node_3466",DataType::kFLOAT,DimsHW{33,192});
    inputs[20]=network->addInput("mobile_origintoken_float32",DataType::kFLOAT,Dims{1,{12},{DimensionType::kCHANNEL}});
    inputs[21]=network->addInput("Node_4187",DataType::kFLOAT,DimsHW{12,192});
    inputs[22]=network->addInput("Node_4203",DataType::kFLOAT,DimsHW{12,192});
    inputs[23]=network->addInput("Node_4219",DataType::kFLOAT,DimsHW{12,192});
    inputs[24]=network->addInput("umid_origintoken_input_id_int32",DataType::kINT32,Dims{1,{133},{DimensionType::kCHANNEL}});
    inputs[25]=network->addInput("umid_origintoken_input_mask_float32",DataType::kFLOAT,Dims{1,{133},{DimensionType::kCHANNEL}});
    inputs[26]=network->addInput("umid_typetoken_int32",DataType::kINT32,Dims{1,{133},{DimensionType::kCHANNEL}});
    inputs[27]=network->addInput("umid_relativepositiontoken_int32",DataType::kINT32,Dims{1,{133},{DimensionType::kCHANNEL}});
    inputs[28]=network->addInput("email_origintoken_input_id_int32",DataType::kINT32,Dims{1,{35},{DimensionType::kCHANNEL}});
    inputs[29]=network->addInput("email_origintoken_input_mask_float32",DataType::kFLOAT,Dims{1,{35},{DimensionType::kCHANNEL}});
    inputs[30]=network->addInput("email_typetoken_int32",DataType::kINT32,Dims{1,{35},{DimensionType::kCHANNEL}});
    inputs[31]=network->addInput("email_relativepositiontoken_int32",DataType::kINT32,Dims{1,{35},{DimensionType::kCHANNEL}});
    inputs[32]=network->addInput("event_origintoken_input_id_int32",DataType::kINT32,Dims{1,{16},{DimensionType::kCHANNEL}});
    inputs[33]=network->addInput("event_origintoken_input_mask_float32",DataType::kFLOAT,Dims{1,{16},{DimensionType::kCHANNEL}});
    inputs[34]=network->addInput("event_typetoken_int32",DataType::kINT32,Dims{1,{16},{DimensionType::kCHANNEL}});
    inputs[35]=network->addInput("event_relativepositiontoken_int32",DataType::kINT32,Dims{1,{16},{DimensionType::kCHANNEL}});

    //build the sample
    if (!sample.build(builder,network,config,inputs,outputs,batch_size,dlr_logger)){
      dlr_logger.log(ILogger::Severity::kERROR,"Failed to build the model!");
      return -1;
    }

    //mark output tensors
    network->markOutput(*outputs[0]);
    network->markOutput(*outputs[1]);

    //serialize engine to file and read from file
    bool serialized=serialize_engine_to_file("ssl_login_get_emb_1.trt",builder,network,config,dlr_logger);
    if (!serialized){
      dlr_logger.log(ILogger::Severity::kERROR,"Serialize failed!");
      return -1;
    }
    print_center("Engine of type float32 serialized to ssl_login_get_emb_1.trt");
  }

  //load the engine from file
  bool deserialized=deserialize_engine_from_file("ssl_login_get_emb_1.trt",engine,dlr_logger);
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
  /*
  if(!show_detail){
    print_center("<Start> Test ssl_login_get_emb_1 testset");
    passed=test(engine,batch_size,show_detail);
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" <End> Test ssl_login_get_emb_1 testset");
    std::cout<<std::endl;
  }
  if(passed){
    print_center("<Start> Test ssl_login_get_emb_1 inference");
    double totalTime;
    passed=infer(engine,batch_size,repeat_num,show_detail,totalTime);
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" <End> Test ssl_login_get_emb_1 inference");
    std::cout<<std::endl;
    //report QPS
    double avgTime = double(totalSuccessTimeUsed)/(repeat_num*1000);
    double QPS = totalSuccessNum*1000000/totalTime;
    std::cout<<"Batch size "<<batch_size<<", Repeat num "<<repeat_num<<" -> avgTime : "<<avgTime<<" ms, QPS : "<<QPS<<std::endl;
    std::cerr<<"[RESULTS] QPS : "<<QPS<<std::endl;
  }
  */
  //clean up the tools
  sample.clean_up();
  return passed ? 0:-1;
}
