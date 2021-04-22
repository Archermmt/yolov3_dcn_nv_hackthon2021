//
// Created by chengjin on 2020-06-02.
//

#include "trt_utils.h"
#include <chrono>

using namespace nvinfer1;
using namespace std;

namespace quake {
namespace framework {
namespace ops_lib {

void show_output_shape(ILayer* layer,int id){
  Dims dims=layer->getOutput(id)->getDimensions();
  std::cout<<"Shape of "<<id<<" th output: ";
  for(int i=0;i<dims.nbDims;i++){
    std::cout<<dims.d[i]<<',';
  }
  std::cout<<std::endl;
}

std::map<std::string,Weights> load_weigths(const std::string& file){
  std::map<std::string,Weights> weightMap;
  // Open weights file
  std::ifstream input(file,std::ios::binary);
  assert(input.is_open() && ("Failed to open file "+file).c_str());

  // Read number of weight blobs
  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");
  std::cout<<"Find "<<count<<" weigths in the file : "<<file<<std::endl;
  
  while (count--)
  {
    Weights wt{DataType::kFLOAT, nullptr, 0};
    uint32_t type, size;
    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> type >> size;
    wt.type = static_cast<DataType>(type);

    // Load blob
    if (wt.type == DataType::kFLOAT)
    {
      uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
      for (uint32_t x = 0; x < size; ++x)
      {
        input >> std::hex >> val[x];
      }
      wt.values = val;
    }
    else if (wt.type == DataType::kHALF)
    {
      uint16_t* val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
      for (uint32_t x = 0; x < size; ++x)
      {
        input >> std::hex >> val[x];
      }
      wt.values = val;
    }
    wt.count = size;
    weightMap[name] = wt;
  }
  input.close();
  return weightMap;
}

bool clean_weights(std::map<std::string, Weights> weightMap){
  // Clean up the weights
  for (auto& mem : weightMap)
  {
    free((void*) (mem.second.values));
  }
  return true;
}

size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

void* copyToDevice(const void* data, size_t count){
  void* deviceData;
  CHECK(cudaMalloc(&deviceData, count));
  CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
  return deviceData;
}

void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size){
  deviceWeights = copyToDevice(hostBuffer, size);
  hostBuffer += size;
}

void convertAndCopyToDevice(void*& deviceWeights, const Weights& weights, DataType data_type){
  if (weights.type != data_type) {
    // Weights are converted in host memory first, if the type does not match
    size_t size = weights.count * (data_type == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
    void* buffer = malloc(size);
    for (int64_t v = 0; v < weights.count; ++v)
        if (data_type == DataType::kFLOAT)
            static_cast<float*>(buffer)[v] = dlr_half2float(static_cast<const __half*>(weights.values)[v]);
        else
            static_cast<__half*>(buffer)[v] = dlr_float2half(static_cast<const float*>(weights.values)[v]);

    deviceWeights = copyToDevice(buffer, size);
    free(buffer);
  } else {
    deviceWeights = copyToDevice(weights.values, weights.count * type2size(data_type));
  }
}

void convertAndCopyToBuffer(char*& buffer, const Weights& weights, DataType data_type){
  if (weights.type != data_type)
    for (int64_t v = 0; v < weights.count; ++v)
      if (data_type == DataType::kFLOAT)
        reinterpret_cast<float*>(buffer)[v] = dlr_half2float(static_cast<const __half*>(weights.values)[v]);
      else
        reinterpret_cast<__half*>(buffer)[v] = dlr_float2half(static_cast<const float*>(weights.values)[v]);
  else
    memcpy(buffer, weights.values, weights.count * type2size(data_type));
  buffer += weights.count * type2size(data_type);
}
/*
bool serialize_engine_to_file(const std::string& file,
  TRTUniquePtr<IBuilder>& builder,
  TRTUniquePtr<INetworkDefinition>& network,
  DLRLogger& logger){
  auto engine=TRTUniquePtr<ICudaEngine>(builder->buildCudaEngine(*network));
  if (!engine){
    logger.log(ILogger::Severity::kERROR,"create engine failed!");
    return false;
  }
  auto serializedModel=TRTUniquePtr<IHostMemory>(engine->serialize());
  std::ofstream ofs(file,std::ios::out | std::ios::binary);
  assert(ofs.is_open() && ("Failed to open file "+file).c_str());
  //write to file
  ofs.write((char*)(serializedModel->data()),serializedModel->size());
  ofs.close();
  logger.log(ILogger::Severity::kINFO,("serialize engine to "+file).c_str());  
  return true;
}
*/
bool serialize_engine_to_file(const std::string& file,
  TRTUniquePtr<IBuilder>& builder,
  TRTUniquePtr<INetworkDefinition>& network,
  TRTUniquePtr<IBuilderConfig>& config,
  DLRLogger& logger){
  std::cout<<"calling serialize to engine "<<std::endl;

  std::chrono::steady_clock::time_point build_begin=std::chrono::steady_clock::now();
  auto engine=TRTUniquePtr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
  if (!engine){
    logger.log(ILogger::Severity::kERROR,"create engine failed!");
    return false;
  }
  std::chrono::steady_clock::time_point build_end=std::chrono::steady_clock::now();
  double build_time = std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_begin).count();
  std::cout<<"[INFO] Build engine with "<<build_time/1000000<<" s"<<std::endl;;

  auto serializedModel=TRTUniquePtr<IHostMemory>(engine->serialize());
  std::chrono::steady_clock::time_point serialize_end=std::chrono::steady_clock::now();
  double serialize_time = std::chrono::duration_cast<std::chrono::microseconds>(serialize_end - build_end).count();
  std::cout<<"[INFO] Serialize engine with "<<serialize_time/1000000<<" s"<<std::endl;

  std::ofstream ofs(file,std::ios::out | std::ios::binary);
  assert(ofs.is_open() && ("Failed to open file "+file).c_str());
  //write to file
  ofs.write((char*)(serializedModel->data()),serializedModel->size());

  ofs.close();
  logger.log(ILogger::Severity::kINFO,("serialize engine to "+file).c_str());  
  //clean up
  return true;
}

bool deserialize_engine_from_file(const std::string& file,std::shared_ptr<ICudaEngine>& engine,DLRLogger& logger){
  std::vector<char> stream;
  size_t size{0};
  std::ifstream input(file,std::ifstream::binary);
  assert(input.is_open() && ("Failed to open file "+file).c_str());
  if (input.good()){
    input.seekg(0, input.end);
    size = input.tellg();
    input.seekg(0, input.beg);
    stream.resize(size);
    input.read(stream.data(), size);
    input.close();
  }
  logger.log(ILogger::Severity::kINFO,("size of engine from "+file+" is "+std::to_string(size)).c_str());
  auto runtime=TRTUniquePtr<IRuntime>(createInferRuntime(logger));
  engine=std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(stream.data(), size, nullptr),InferDeleter());
  input.close();
  return true;
}

void enableDLA(TRTUniquePtr<IBuilder>& builder, TRTUniquePtr<IBuilderConfig>& config, int useDLACore, bool allowGPUFallback){
  if (useDLACore >= 0){
    if (builder->getNbDLACores() == 0){
      std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores" << std::endl;
      assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
    }
    if (allowGPUFallback){
      config->setFlag(BuilderFlag::kGPU_FALLBACK);
    }
    if (!builder->getInt8Mode() && !config->getFlag(BuilderFlag::kINT8)){
      // User has not requested INT8 Mode.
      // By default run in FP16 mode. FP32 mode is not permitted.
      builder->setFp16Mode(true);
      config->setFlag(BuilderFlag::kFP16);
    }
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(useDLACore);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
  }
}

DLRBatchStream::DLRBatchStream(const std::string& batchPath,int batchSize,int maxBatches,const std::string& info_file)
  : mBatchPath(batchPath)
  , mBatchSize(batchSize) 
  , mMaxBatches(maxBatches)
{
  //read tensor size
  std::ifstream in_info(info_file,std::ios::binary);
  assert(in_info.is_open() && ("Failed to open file "+info_file).c_str());
  std::string line;
  while (std::getline(in_info,line)){
    int pos=line.find(":");
    std::string tensorName=line.substr(0,pos);
    size_t size=std::stoi(line.substr(pos+1));
    mTensorNames.emplace_back(tensorName);
    mTensorSizes.emplace_back(size);
  }
  //check if all name files exist
  for(unsigned int i=0;i<mTensorNames.size();i++){
    std::string batchfile = mBatchPath+"/"+mTensorNames[i]+"/batch_0.bin";
    FILE* file = fopen(batchfile.c_str(), "rb");
    assert(file!=NULL && ("Failed to open file "+batchfile).c_str());
    mBatchStride+=mTensorSizes[i];
  }
  //malloc batch size
  mBatch.resize(mBatchSize*mBatchStride,0);
  reset(0);
}

void DLRBatchStream::reset(int firstBatch)
{
  mBatchCount = 0;
  mStartTiming = false;
  skip(firstBatch);
}

bool DLRBatchStream::next()
{
  if (mBatchCount == mMaxBatches){
    std::cout<<"[DLR_INFO] Reach maxbatches: "<<mMaxBatches<<std::endl;
    return false;
  }

  bool data_loaded=true;
  int pos=0;
  for(unsigned int t_id=0;t_id<mTensorNames.size();t_id++){
    #pragma omp parallel for
    for (int batchPos=0;batchPos<mBatchSize;batchPos++){
      std::string batchfile = mBatchPath+"/"+mTensorNames[t_id]+"/batch_"+std::to_string(mFileCount+batchPos)+".bin";
      bool success=FileUtils::read_file_to_buffer(batchfile,mBatch.data()+pos+batchPos*mTensorSizes[t_id],mTensorSizes[t_id],false);
      if(!success){
        //std::cout<<"[DLR_INFO] Failed to read at file count : "<<mFileCount+batchPos<<", end up the feeding!"<<std::endl;
        mBatchCount=mMaxBatches;
        data_loaded=false;
      }
    }
    pos+=mBatchSize*mTensorSizes[t_id];
  }
  if(data_loaded){
    double avg_time=0;
    if(!mStartTiming || mBatchCount<=0){
      mStartPoint=clock();
      mStartTiming=true;
    }else{
      clock_t cur_time=clock();
      avg_time=(cur_time-mStartPoint)*1.0/(CLOCKS_PER_SEC*(mBatchCount));
    }
    std::cout<<"[DLR_INFO] Getting batch "<<mBatchCount*mBatchSize<<":"<<(mBatchCount+1)*mBatchSize<<"/"<<(mMaxBatches>0 ? mMaxBatches*mBatchSize : -1)<<", avg "<<avg_time*1000<<" ms/batch"<<std::endl;
    mFileCount+=mBatchSize;
    mBatchCount++;
  }else{
    std::cout<<"[DLR_INFO] Failed to read at : "<<mFileCount<<", end up the feeding!"<<std::endl;
  }
  return data_loaded;
}

void DLRBatchStream::skip(int skipCount)
{
  assert(mMaxBatches==-1 || skipCount<mMaxBatches);
  int x = mBatchCount;
  for (int i = 0; i < skipCount; i++){
    next();
    mFileCount+=mBatchSize;
  }
  mBatchCount = x;
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
