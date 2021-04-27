#include "yolov3_dcn_1.h"
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
  void* cpu_buffers[4];
  void* gpu_buffers[4];

  // Malloc inputs buffers
  const int input_0 = engine->getBindingIndex("image");
  CHECK(cudaMalloc(&gpu_buffers[input_0],batch_size*1108992*sizeof(float)));
  cpu_buffers[input_0]=malloc(batch_size*1108992*sizeof(float));
  const int input_1 = engine->getBindingIndex("im_size_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_1],batch_size*2*sizeof(float)));
  cpu_buffers[input_1]=malloc(batch_size*2*sizeof(float));

  // Malloc the output buffers
  const int output_0 = engine->getBindingIndex("multinms");
  CHECK(cudaMalloc(&gpu_buffers[output_0],batch_size*40000*sizeof(float)));
  cpu_buffers[output_0]=malloc(batch_size*40000*sizeof(float));
  const int output_1 = engine->getBindingIndex("multiclass_nms_0_1");
  CHECK(cudaMalloc(&gpu_buffers[output_1],batch_size*80*sizeof(float)));
  cpu_buffers[output_1]=malloc(batch_size*80*sizeof(float));

  // Stream obj for data feeding
  DLRBatchStream data_stream("/usr/local/quake/datas/testset/yolov3_dcn",batch_size,-1,"yolov3_dcn_1.tensor_info");
  while (data_stream.next()){
    void* data=data_stream.getBatch();
    int pos=0;
    // memcopy from host to device
    CHECK(cudaMemcpyAsync(gpu_buffers[input_0],data+pos,batch_size*1108992*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*1108992*sizeof(float);
    CHECK(cudaMemcpyAsync(gpu_buffers[input_1],data+pos,batch_size*2*sizeof(float),cudaMemcpyHostToDevice,stream));
    pos+=batch_size*2*sizeof(float);
    cudaStreamSynchronize(stream);

    context->enqueue(batch_size,gpu_buffers,stream,nullptr);

    // memcopy from device to host
    CHECK(cudaMemcpyAsync(cpu_buffers[output_0],gpu_buffers[output_0],batch_size*40000*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_1],gpu_buffers[output_1],batch_size*80*sizeof(float),cudaMemcpyDeviceToHost,stream));
    // verify the outputs
    failed_cnt=verify_buffer("multinms_batch_"+std::to_string(data_stream.getBatchesRead()),(float*)(data+pos),(float*)cpu_buffers[output_0],batch_size*40000,0.05*100,show_detail);
    if(failed_cnt>batch_size*40000*0.05)
      passed=false;
    pos+=batch_size*40000*sizeof(float);
    failed_cnt=verify_buffer("multiclass_nms_0_1_batch_"+std::to_string(data_stream.getBatchesRead()),(float*)(data+pos),(float*)cpu_buffers[output_1],batch_size*80,0.05*100,show_detail);
    if(failed_cnt>batch_size*80*0.05)
      passed=false;
    pos+=batch_size*80*sizeof(float);
    // check results
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" Batch "+std::to_string(data_stream.getBatchesRead()));
  }

  // Release stream and device buffers
  cudaStreamDestroy(stream);
  for (int i=0;i<4;i++){
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
  void* cpu_buffers[4];
  void* gpu_buffers[4];

  // Malloc inputs buffers
  const int input_0 = engine->getBindingIndex("image");
  CHECK(cudaMalloc(&gpu_buffers[input_0],batch_size*1108992*sizeof(float)));
  cpu_buffers[input_0]=malloc(batch_size*1108992*sizeof(float));
  const int input_1 = engine->getBindingIndex("im_size_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_1],batch_size*2*sizeof(float)));
  cpu_buffers[input_1]=malloc(batch_size*2*sizeof(float));

  // Malloc the output buffers
  const int output_0 = engine->getBindingIndex("multinms");
  CHECK(cudaMalloc(&gpu_buffers[output_0],batch_size*40000*sizeof(float)));
  cpu_buffers[output_0]=malloc(batch_size*40000*sizeof(float));
  const int output_1 = engine->getBindingIndex("multiclass_nms_0_1");
  CHECK(cudaMalloc(&gpu_buffers[output_1],batch_size*80*sizeof(float)));
  cpu_buffers[output_1]=malloc(batch_size*80*sizeof(float));

  // Copy input data from file to CPU buffers
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/yolov3_dcn/image.bin",(float*)cpu_buffers[input_0],batch_size*1108992,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/yolov3_dcn/im_size_float32.bin",(float*)cpu_buffers[input_1],batch_size*2,show_detail);

  // Enqueue the task and record time
  std::chrono::steady_clock::time_point test_begin=std::chrono::steady_clock::now();
  for(int i=0;i<repeat_num;i++){
    std::chrono::steady_clock::time_point begin=std::chrono::steady_clock::now();
    // memcopy from host to device
    CHECK(cudaMemcpyAsync(gpu_buffers[input_0],cpu_buffers[input_0],batch_size*1108992*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_1],cpu_buffers[input_1],batch_size*2*sizeof(float),cudaMemcpyHostToDevice,stream));
    cudaStreamSynchronize(stream);

    context->enqueue(batch_size,gpu_buffers,stream,nullptr);

    // memcopy from device to host
    CHECK(cudaMemcpyAsync(cpu_buffers[output_0],gpu_buffers[output_0],batch_size*40000*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_1],gpu_buffers[output_1],batch_size*80*sizeof(float),cudaMemcpyDeviceToHost,stream));
    cudaStreamSynchronize(stream);
    std::chrono::steady_clock::time_point end=std::chrono::steady_clock::now();
    totalSuccessNum+=batch_size;
    totalSuccessTimeUsed+=std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
  }
  std::chrono::steady_clock::time_point test_end=std::chrono::steady_clock::now();
  totalTime = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_begin).count();

  //Verify the outputs
  failed_cnt=FileUtils::verify_buffer_with_file("multinms","/usr/local/quake/datas/inference_datas/yolov3_dcn/multinms.bin",(float*)cpu_buffers[output_0],batch_size*40000,0.05*100,show_detail);
  if(failed_cnt>batch_size*40000*0.05)
    passed=false;

  failed_cnt=FileUtils::verify_buffer_with_file("multiclass_nms_0_1","/usr/local/quake/datas/inference_datas/yolov3_dcn//multiclass_nms_0_1.bin",(float*)cpu_buffers[output_1],batch_size*80,0.05*100,show_detail);
  if(failed_cnt>batch_size*80*0.05)
    passed=false;

  // Release stream and device buffers
  cudaStreamDestroy(stream);
  for (int i=0;i<4;i++){
    CHECK(cudaFree(gpu_buffers[i]));
    free(cpu_buffers[i]);
  }
  return passed;
}

int main(int argc, char** argv){
  dlr_logger.log(ILogger::Severity::kINFO,"Building and running a GPU inference engine for yolov3_dcn_1");
  yolov3_dcn_1 sample;

  bool show_detail=false;
  int batch_size=1;
  int repeat_num=1;

  if(!show_detail){
    batch_size=argc>1? atoi(argv[1]):1;
    repeat_num=argc>2? atoi(argv[2]):50;
  }

  //define the engine
  std::shared_ptr<ICudaEngine> engine;

  if(!FileUtils::file_exist("yolov3_dcn_1.trt")){
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
    ITensor* inputs[2];
    ITensor* outputs[2];

    inputs[0]=network->addInput("image",DataType::kFLOAT,Dims3{3,608,608});
    inputs[1]=network->addInput("im_size_float32",DataType::kFLOAT,Dims{1,{2},{DimensionType::kCHANNEL}});

    //build the sample
    if (!sample.build(builder,network,config,inputs,outputs,batch_size,dlr_logger)){
      dlr_logger.log(ILogger::Severity::kERROR,"Failed to build the model!");
      return -1;
    }

    //mark output tensors
    network->markOutput(*outputs[0]);
    network->markOutput(*outputs[1]);

    //serialize engine to file and read from file
    bool serialized=serialize_engine_to_file("yolov3_dcn_1.trt",builder,network,config,dlr_logger);
    if (!serialized){
      dlr_logger.log(ILogger::Severity::kERROR,"Serialize failed!");
      return -1;
    }
    print_center("Engine of type float32 serialized to yolov3_dcn_1.trt");
  }

  //load the engine from file
  bool deserialized=deserialize_engine_from_file("yolov3_dcn_1.trt",engine,dlr_logger);
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
    print_center("<Start> Test yolov3_dcn_1 testset");
    passed=test(engine,batch_size,show_detail);
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" <End> Test yolov3_dcn_1 testset");
    std::cout<<std::endl;
  }*/
  if(passed){
    print_center("<Start> Test yolov3_dcn_1 inference");
    double totalTime;
    passed=infer(engine,batch_size,repeat_num,show_detail,totalTime);
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" <End> Test yolov3_dcn_1 inference");
    std::cout<<std::endl;
    //report QPS
    double avgTime = double(totalSuccessTimeUsed)/(repeat_num*1000);
    double QPS = totalSuccessNum*1000000/totalTime;
    std::cout<<"Batch size "<<batch_size<<", Repeat num "<<repeat_num<<" -> avgTime : "<<avgTime<<" ms, QPS : "<<QPS<<std::endl;
    std::cerr<<"[RESULTS] QPS : "<<QPS<<std::endl;
    return 0;
  }

  //clean up the tools
  sample.clean_up();
  return 0;
  //return passed ? 0:-1;
}
