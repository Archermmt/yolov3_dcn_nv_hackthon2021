//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_TRT_UTILS_H
#define OPSLIB_TRT_UTILS_H

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <memory>
#include <time.h>
#include "util/file_utils.h"
#include "kernel/gpu/cu_utils.h"

using namespace nvinfer1;
using namespace std;

namespace quake {
namespace framework {
namespace ops_lib {

#define CHECK(status)                                    \
  do                                                     \
  {                                                      \
    auto ret = (status);                                 \
    if (ret != 0)                                        \
    {                                                    \
      std::cout << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)


class DLRLogger : public ILogger{  
  public:
    void set_verbose_level(int level){
      mVerboseLevel=level;
    }
    void log(Severity severity, const char* msg) override
    {
      time_t rawtime;
      struct tm * timeinfo;
      time( &rawtime );
      timeinfo = localtime( &rawtime );
      std::string time_str=std::to_string(timeinfo->tm_hour)+":"+std::to_string(timeinfo->tm_min)+":"+std::to_string(timeinfo->tm_sec)+" ";
      switch(severity)
      {
        case Severity::kINTERNAL_ERROR:
          if(mVerboseLevel>=0)
            std::cout <<time_str<<"[TensorRT.INTERNAL_ERROR]: " <<msg<<std::endl;
          break;
        case Severity::kERROR:
          if(mVerboseLevel>=0)
            std::cout <<time_str<<"[TensorRT.ERROR]: " <<msg<<std::endl;
          break;
        case Severity::kWARNING:
          if(mVerboseLevel>=1)
            std::cout <<time_str<<"[TensorRT.WARNING]: " <<msg<<std::endl;
          break;
        case Severity::kINFO:
          if(mVerboseLevel>=1)
            std::cout <<time_str<<"[TensorRT.INFO]: " <<msg<<std::endl;;
          break;
        default:
          if(mVerboseLevel>=2)
            std::cout <<time_str<<"[TensorRT.DEBUG]: " <<msg<<std::endl;;
          break;
      }
    }
  private:
    int mVerboseLevel{0};
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T,InferDeleter>;

void show_output_shape(ILayer* layer,int id=0);

std::map<std::string,Weights> load_weigths(const std::string& file);

bool clean_weights(std::map<std::string, Weights> weightMap);

size_t type2size(DataType type);

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val){
  *reinterpret_cast<T*>(buffer) = val;
  buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
void readFromBuffer(const char*& buffer, T& val){
  val = *reinterpret_cast<const T*>(buffer);
  buffer += sizeof(T);
}

void* copyToDevice(const void* data, size_t count);

void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size);

void convertAndCopyToDevice(void*& deviceWeights, const Weights& weights, DataType data_type);

void convertAndCopyToBuffer(char*& buffer, const Weights& weights, DataType data_type);

bool serialize_engine_to_file(const std::string& file,TRTUniquePtr<IBuilder>& builder,
  TRTUniquePtr<INetworkDefinition>& network,TRTUniquePtr<IBuilderConfig>& config,DLRLogger& logger);

bool deserialize_engine_from_file(const std::string& file,std::shared_ptr<ICudaEngine>& engine,DLRLogger& logger);

void enableDLA(TRTUniquePtr<IBuilder>& builder,TRTUniquePtr<IBuilderConfig>& config, int useDLACore, bool allowGPUFallback);

class DLRBatchStream
{
public:
  DLRBatchStream(const std::string& batchPath,int batchSize,int maxBatches,const std::string& info_file="tensor_info");

  void reset(int firstBatch);

  bool next();

  void skip(int skipCount);

  void *getBatch() {return (void*)mBatch.data();}
  std::string getBatchPath() const {return mBatchPath;}
  int getBatchesRead() const {return mBatchCount;}
  int getBatchSize() const {return mBatchSize;}
  int getBatchStride() const {return mBatchStride;}
  int getFileCount() const {return mFileCount;}
  std::string tname_at(int id) {return mTensorNames[id];}
  int tsize_at(int id) {return mTensorSizes[id];}
  
private:
  std::string mBatchPath;
  int mBatchSize{0};
  int mMaxBatches{0};
  int mFileCount{0};
  int mBatchCount{0};
  int mBatchStride{0};
  bool mStartTiming{false};
  clock_t mStartPoint;

  std::vector<char> mBatch;
  std::vector<std::string> mTensorNames;
  std::vector<size_t> mTensorSizes;
};

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_TRT_UTILS_H
