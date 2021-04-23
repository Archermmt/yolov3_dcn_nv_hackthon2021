#include "test_1.h"
#include "util/file_utils.h"
#include "util/base.h"
#include <chrono>
#include <atomic>

using namespace quake::framework::ops_lib;
DLRLogger dlr_logger=DLRLogger();

std::atomic<long> totalSuccessNum(0L);
std::atomic<long> totalSuccessTimeUsed(0L);

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
  void* cpu_buffers[5];
  void* gpu_buffers[5];

  // Malloc inputs buffers
  const int input_0 = engine->getBindingIndex("input_ids_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_0],batch_size*256*sizeof(int)));
  cpu_buffers[input_0]=malloc(batch_size*256*sizeof(int));
  const int input_1 = engine->getBindingIndex("input_mask_float32");
  CHECK(cudaMalloc(&gpu_buffers[input_1],batch_size*256*sizeof(float)));
  cpu_buffers[input_1]=malloc(batch_size*256*sizeof(float));
  const int input_2 = engine->getBindingIndex("segment_ids_int32");
  CHECK(cudaMalloc(&gpu_buffers[input_2],batch_size*256*sizeof(int)));
  cpu_buffers[input_2]=malloc(batch_size*256*sizeof(int));
  const int input_3 = engine->getBindingIndex("augment_img");
  CHECK(cudaMalloc(&gpu_buffers[input_3],batch_size*150528*sizeof(float)));
  cpu_buffers[input_3]=malloc(batch_size*150528*sizeof(float));

  // Malloc the output buffers
  const int output_0 = engine->getBindingIndex("predscore");
  CHECK(cudaMalloc(&gpu_buffers[output_0],batch_size*2*sizeof(float)));
  cpu_buffers[output_0]=malloc(batch_size*2*sizeof(float));

  // Copy input data from file to CPU buffers
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/const_shape/input_ids_int32.bin",(int*)cpu_buffers[input_0],batch_size*256,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/const_shape/input_mask_float32.bin",(float*)cpu_buffers[input_1],batch_size*256,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/const_shape/segment_ids_int32.bin",(int*)cpu_buffers[input_2],batch_size*256,show_detail);
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/const_shape/augment_img.bin",(float*)cpu_buffers[input_3],batch_size*150528,show_detail);

  // Enqueue the task and record time
  std::chrono::steady_clock::time_point test_begin=std::chrono::steady_clock::now();
  for(int i=0;i<repeat_num;i++){
    std::chrono::steady_clock::time_point begin=std::chrono::steady_clock::now();
    // memcopy from host to device
    CHECK(cudaMemcpyAsync(gpu_buffers[input_0],cpu_buffers[input_0],batch_size*256*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_1],cpu_buffers[input_1],batch_size*256*sizeof(float),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_2],cpu_buffers[input_2],batch_size*256*sizeof(int),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(gpu_buffers[input_3],cpu_buffers[input_3],batch_size*150528*sizeof(float),cudaMemcpyHostToDevice,stream));
    cudaStreamSynchronize(stream);

    context->enqueue(batch_size,gpu_buffers,stream,nullptr);

    // memcopy from device to host
    CHECK(cudaMemcpyAsync(cpu_buffers[output_0],gpu_buffers[output_0],batch_size*2*sizeof(float),cudaMemcpyDeviceToHost,stream));
    cudaStreamSynchronize(stream);
    std::chrono::steady_clock::time_point end=std::chrono::steady_clock::now();
    totalSuccessNum+=batch_size;
    totalSuccessTimeUsed+=std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
  }
  std::chrono::steady_clock::time_point test_end=std::chrono::steady_clock::now();
  totalTime = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_begin).count();

  //Verify the outputs
  failed_cnt=FileUtils::verify_buffer_with_file("predscore","/usr/local/quake/datas/inference_datas/const_shape/predscore.bin",(float*)cpu_buffers[output_0],batch_size*2,0.1*100,show_detail);
  if(failed_cnt>batch_size*2*0.1)
    passed=false;

  // Release stream and device buffers
  cudaStreamDestroy(stream);
  for (int i=0;i<5;i++){
    CHECK(cudaFree(gpu_buffers[i]));
    free(cpu_buffers[i]);
  }
  return passed;
}

int main(int argc, char** argv){
  dlr_logger.log(ILogger::Severity::kINFO,"Building and running a GPU inference engine for test_1");
  test_1 sample;

  bool show_detail=false;
  int batch_size=1;
  int repeat_num=1;

  if(!show_detail){
    batch_size=argc>1? atoi(argv[1]):8;
    repeat_num=argc>2? atoi(argv[2]):1000;
  }

  //define the engine
  std::shared_ptr<ICudaEngine> engine;

  if(!FileUtils::file_exist("test_1.trt")){
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
    ITensor* inputs[4];
    ITensor* outputs[1];

    inputs[0]=network->addInput("input_ids_int32",DataType::kINT32,Dims{1,{256},{DimensionType::kCHANNEL}});
    inputs[1]=network->addInput("input_mask_float32",DataType::kFLOAT,Dims{1,{256},{DimensionType::kCHANNEL}});
    inputs[2]=network->addInput("segment_ids_int32",DataType::kINT32,Dims{1,{256},{DimensionType::kCHANNEL}});
    inputs[3]=network->addInput("augment_img",DataType::kFLOAT,Dims3{3,224,224});

    //build the sample
    if (!sample.build(builder,network,config,inputs,outputs,batch_size,dlr_logger)){
      dlr_logger.log(ILogger::Severity::kERROR,"Failed to build the model!");
      return -1;
    }

    //mark output tensors
    network->markOutput(*outputs[0]);

    //serialize engine to file and read from file
    bool serialized=serialize_engine_to_file("test_1.trt",builder,network,config,dlr_logger);
    if (!serialized){
      dlr_logger.log(ILogger::Severity::kERROR,"Serialize failed!");
      return -1;
    }
    print_center("Engine of type float32 serialized to test_1.trt");
  }

  //load the engine from file
  bool deserialized=deserialize_engine_from_file("test_1.trt",engine,dlr_logger);
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

  if(passed){
    print_center("<Start> Test test_1 inference");
    double totalTime;
    passed=infer(engine,batch_size,repeat_num,show_detail,totalTime);
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" <End> Test test_1 inference");
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
