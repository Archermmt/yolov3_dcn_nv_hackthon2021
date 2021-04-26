#include "ssha_1.h"
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
  void* cpu_buffers[6];
  void* gpu_buffers[6];

  // Malloc inputs buffers
  const int input_0 = engine->getBindingIndex("data");
  CHECK(cudaMalloc(&gpu_buffers[input_0],batch_size*1080000*sizeof(char)));
  cpu_buffers[input_0]=malloc(batch_size*1080000*sizeof(char));

  // Malloc the output buffers
  const int output_0 = engine->getBindingIndex("reshape4");
  CHECK(cudaMalloc(&gpu_buffers[output_0],batch_size*2888*sizeof(float)));
  cpu_buffers[output_0]=malloc(batch_size*2888*sizeof(float));
  const int output_1 = engine->getBindingIndex("reshape5");
  CHECK(cudaMalloc(&gpu_buffers[output_1],batch_size*722*sizeof(float)));
  cpu_buffers[output_1]=malloc(batch_size*722*sizeof(float));
  const int output_2 = engine->getBindingIndex("reshape2");
  CHECK(cudaMalloc(&gpu_buffers[output_2],batch_size*11552*sizeof(float)));
  cpu_buffers[output_2]=malloc(batch_size*11552*sizeof(float));
  const int output_3 = engine->getBindingIndex("reshape3");
  CHECK(cudaMalloc(&gpu_buffers[output_3],batch_size*2888*sizeof(float)));
  cpu_buffers[output_3]=malloc(batch_size*2888*sizeof(float));
  const int output_4 = engine->getBindingIndex("_plus8");
  CHECK(cudaMalloc(&gpu_buffers[output_4],batch_size*720000*sizeof(float)));
  cpu_buffers[output_4]=malloc(batch_size*720000*sizeof(float));

  // Copy input data from file to CPU buffers
  FileUtils::read_file_to_buffer("/usr/local/quake/datas/inference_datas/out_mismatch/data.bin",(char*)cpu_buffers[input_0],batch_size*1080000,show_detail);

  // Enqueue the task and record time
  std::chrono::steady_clock::time_point test_begin=std::chrono::steady_clock::now();
  for(int i=0;i<repeat_num;i++){
    std::chrono::steady_clock::time_point begin=std::chrono::steady_clock::now();
    // memcopy from host to device
    CHECK(cudaMemcpyAsync(gpu_buffers[input_0],cpu_buffers[input_0],batch_size*1080000*sizeof(char),cudaMemcpyHostToDevice,stream));
    cudaStreamSynchronize(stream);

    context->enqueue(batch_size,gpu_buffers,stream,nullptr);

    // memcopy from device to host
    CHECK(cudaMemcpyAsync(cpu_buffers[output_0],gpu_buffers[output_0],batch_size*2888*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_1],gpu_buffers[output_1],batch_size*722*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_2],gpu_buffers[output_2],batch_size*11552*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_3],gpu_buffers[output_3],batch_size*2888*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(cpu_buffers[output_4],gpu_buffers[output_4],batch_size*720000*sizeof(float),cudaMemcpyDeviceToHost,stream));
    cudaStreamSynchronize(stream);
    std::chrono::steady_clock::time_point end=std::chrono::steady_clock::now();
    totalSuccessNum+=batch_size;
    totalSuccessTimeUsed+=std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
  }
  std::chrono::steady_clock::time_point test_end=std::chrono::steady_clock::now();
  totalTime = std::chrono::duration_cast<std::chrono::microseconds>(test_end - test_begin).count();

  //Verify the outputs
  failed_cnt=FileUtils::verify_buffer_with_file("reshape4","/usr/local/quake/datas/inference_datas/out_mismatch/reshape4.bin",(float*)cpu_buffers[output_0],batch_size*2888,0.05*100,show_detail);
  if(failed_cnt>batch_size*2888*0.05)
    passed=false;

  failed_cnt=FileUtils::verify_buffer_with_file("reshape5","/usr/local/quake/datas/inference_datas/out_mismatch/reshape5.bin",(float*)cpu_buffers[output_1],batch_size*722,0.05*100,show_detail);
  if(failed_cnt>batch_size*722*0.05)
    passed=false;

  failed_cnt=FileUtils::verify_buffer_with_file("reshape2","/usr/local/quake/datas/inference_datas/out_mismatch/reshape2.bin",(float*)cpu_buffers[output_2],batch_size*11552,0.05*100,show_detail);
  if(failed_cnt>batch_size*11552*0.05)
    passed=false;

  failed_cnt=FileUtils::verify_buffer_with_file("reshape3","/usr/local/quake/datas/inference_datas/out_mismatch/reshape3.bin",(float*)cpu_buffers[output_3],batch_size*2888,0.05*100,show_detail);
  if(failed_cnt>batch_size*2888*0.05)
    passed=false;

  failed_cnt=FileUtils::verify_buffer_with_file("_plus8","/usr/local/quake/datas/inference_datas/out_mismatch/_plus8.bin",(float*)cpu_buffers[output_4],batch_size*720000,0.05*100,show_detail);
  if(failed_cnt>batch_size*720000*0.05)
    passed=false;

  // Release stream and device buffers
  cudaStreamDestroy(stream);
  for (int i=0;i<6;i++){
    CHECK(cudaFree(gpu_buffers[i]));
    free(cpu_buffers[i]);
  }
  return passed;
}

int main(int argc, char** argv){
  dlr_logger.log(ILogger::Severity::kINFO,"Building and running a GPU inference engine for ssha_1");
  ssha_1 sample;

  bool show_detail=false;
  int batch_size=1;
  int repeat_num=1;

  if(!show_detail){
    batch_size=argc>1? atoi(argv[1]):4;
    repeat_num=argc>2? atoi(argv[2]):1;
  }

  //define the engine
  std::shared_ptr<ICudaEngine> engine;

  if(!FileUtils::file_exist("ssha_1.trt")){
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
    ITensor* inputs[1];
    ITensor* outputs[5];

    inputs[0]=network->addInput("data",DataType::kFLOAT,Dims{1,{270000},{DimensionType::kCHANNEL}});

    //build the sample
    if (!sample.build(builder,network,config,inputs,outputs,batch_size,dlr_logger)){
      dlr_logger.log(ILogger::Severity::kERROR,"Failed to build the model!");
      return -1;
    }

    //mark output tensors
    network->markOutput(*outputs[0]);
    network->markOutput(*outputs[1]);
    network->markOutput(*outputs[2]);
    network->markOutput(*outputs[3]);
    network->markOutput(*outputs[4]);

    //serialize engine to file and read from file
    bool serialized=serialize_engine_to_file("ssha_1.trt",builder,network,config,dlr_logger);
    if (!serialized){
      dlr_logger.log(ILogger::Severity::kERROR,"Serialize failed!");
      return -1;
    }
    print_center("Engine of type float32 serialized to ssha_1.trt");
  }

  //load the engine from file
  bool deserialized=deserialize_engine_from_file("ssha_1.trt",engine,dlr_logger);
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
    print_center("<Start> Test ssha_1 inference");
    double totalTime;
    passed=infer(engine,batch_size,repeat_num,show_detail,totalTime);
    std::string msg=passed ? "[PASS]":"[FAIL]";
    print_center(msg+" <End> Test ssha_1 inference");
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
