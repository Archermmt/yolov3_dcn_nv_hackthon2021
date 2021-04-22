//
// Created by chengjin on 2020-06-02.
//

#include <iterator>
#include <sstream>
#include <omp.h>
#include <algorithm>
#include "trt_quantize.h"

using namespace nvinfer1;
using namespace std;

namespace quake {
namespace framework {
namespace ops_lib {

DLRInt8CalibratorUtils::DLRInt8CalibratorUtils(DLRBatchStream& stream,int firstBatch,const std::string& network_name,bool readCache)
  : mStream(stream)
  , mNetworkName(network_name)
  , mReadCache(readCache)
{
  mInputCount=mStream.getBatchSize()*mStream.getBatchStride();
  CHECK(cudaMalloc(&mDeviceInput,mInputCount*sizeof(float)));
  mStream.reset(firstBatch);
}   

bool DLRInt8CalibratorUtils::get_batch(void* bindings[], const char* names[], int nbBindings)
{
  if (!mStream.next())
    return false;

  CHECK(cudaMemcpy(mDeviceInput,mStream.getBatch(),mInputCount*sizeof(float),cudaMemcpyHostToDevice));
  int pos=0;
  for(int i=0;i<nbBindings;i++){
    assert(!strcmp(names[i],mStream.tname_at(i).c_str()));
    bindings[i]=mDeviceInput+pos*sizeof(float);
    pos+=mStream.getBatchSize()*mStream.tsize_at(i);
  }
  return true;
}

const void* DLRInt8CalibratorUtils::read_calibration_cache(size_t& length)
{
  mCalibrationCache.clear();
  std::ifstream input(calibration_table_name(), std::ios::binary);
  input >> std::noskipws;
  if (mReadCache && input.good())
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

  length = mCalibrationCache.size();
  return length ? &mCalibrationCache[0] : nullptr;
}

void DLRInt8CalibratorUtils::write_calibration_cache(const void* cache, size_t length)
{
  std::ofstream output(calibration_table_name(), std::ios::binary);
  output.write(reinterpret_cast<const char*>(cache), length);
}

DLRInt8LegacyCalibrator::DLRInt8LegacyCalibrator(DLRBatchStream& stream,int firstBatch,const std::string& network_name,
  double cutoff,double quantile,bool readCache)
  : mCutoff(cutoff)
  , mQuantile(quantile)
{
  mCalibratorUtils.reset(new DLRInt8CalibratorUtils(stream,firstBatch,network_name,readCache));
}

const void* DLRInt8LegacyCalibrator::readHistogramCache(size_t& length)
{
    length = mHistogramCache.size();
    return length ? &mHistogramCache[0] : nullptr;
}

void DLRInt8LegacyCalibrator::writeHistogramCache(const void* cache, size_t length)
{
    mHistogramCache.clear();
    std::copy_n(reinterpret_cast<const char*>(cache), length, std::back_inserter(mHistogramCache));
}

void setLayerPrecision(TRTUniquePtr<INetworkDefinition>& network,DataType dtype,DLRLogger& logger)
{
  logger.log(ILogger::Severity::kINFO,"Setting Per Layer Computation Precision");
  std::string dtype_name;
  switch (dtype){
    case DataType::kFLOAT : dtype_name="float";break;
    case DataType::kHALF  : dtype_name="fp16";break;
    case DataType::kINT8  : dtype_name="int8";break;
    case DataType::kINT32 : dtype_name="int32";break;
    default : dtype_name="unknow";
  }
  for (int i = 0; i < network->getNbLayers(); ++i)
  {
    auto layer = network->getLayer(i);
    // set computation precision of the layer
    layer->setPrecision(dtype);
    // set output type of the tensor
    for (int j = 0; j < layer->getNbOutputs(); ++j)
    {
      std::string tensorName = layer->getOutput(j)->getName();
      layer->setOutputType(j,dtype);
      logger.log(ILogger::Severity::kINFO,("Setting precision for tensor "+tensorName+" with dtype "+dtype_name).c_str());
    }
  }
}

void setAllTensorScales(TRTUniquePtr<INetworkDefinition>& network, float inScales = 2.0f, float outScales = 4.0f){
  // Ensure that all layer inputs have a scale.
  for (int i = 0; i < network->getNbLayers(); i++){
    auto layer = network->getLayer(i);
    for (int j = 0; j < layer->getNbInputs(); j++){
      ITensor* input{layer->getInput(j)};
      // Optional inputs are nullptr here and are from RNN layers.
      if (input != nullptr && !input->dynamicRangeIsSet()){
        input->setDynamicRange(-inScales, inScales);
      }
    }
  }

  // Ensure that all layer outputs have a scale.
  // Tensors that are also inputs to layers are ingored here
  // since the previous loop nest assigned scales to them.
  for (int i = 0; i < network->getNbLayers(); i++){
    auto layer = network->getLayer(i);
    for (int j = 0; j < layer->getNbOutputs(); j++){
      ITensor* output{layer->getOutput(j)};
      // Optional outputs are nullptr here and are from RNN layers.
      if (output != nullptr && !output->dynamicRangeIsSet()){
        // Pooling must have the same input and output scales.
        if (layer->getType() == LayerType::kPOOLING){
          output->setDynamicRange(-inScales, inScales);
        } else {
          output->setDynamicRange(-outScales, outScales);
        }
      }
    }
  }
}

bool setDynamicRange(const std::string& file,TRTUniquePtr<INetworkDefinition>& network,DLRLogger& logger)
{
  std::ifstream iDynamicRangeStream(file, std::ios::binary);
  if (!iDynamicRangeStream){
    logger.log(ILogger::Severity::kERROR,("Could not find per tensor scales file: "+file).c_str());
    return false;
  }
  std::map<std::string,float> mPerTensorDynamicRangeMap;
  //fill map for dynamic range
  std::string line;
  uint32_t* range_val = reinterpret_cast<uint32_t*>(malloc(sizeof(range_val)*1));
  while (std::getline(iDynamicRangeStream, line)){
    int pos=line.find(": ");
    if(pos==-1){
      continue;
    }
    std::string tensorName=line.substr(0,pos);
    std::stringstream ss(line.substr(pos+2));
    ss>>std::hex>>range_val[0];
    float *dynamicRange=reinterpret_cast<float*>(range_val);
    mPerTensorDynamicRangeMap[tensorName] = dynamicRange[0];
    logger.log(ILogger::Severity::kINFO,("range of: "+tensorName+" : "+std::to_string(dynamicRange[0])).c_str());
  }
  free(range_val);
  // set dynamic range for network input tensors
  for (int i = 0; i < network->getNbInputs(); ++i)
  {
    string tName = network->getInput(i)->getName();
    if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end()){
      network->getInput(i)->setDynamicRange(-mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName));
    }else{
      logger.log(ILogger::Severity::kERROR,("Could not find "+tName+" from file, use default 127").c_str());
      network->getInput(i)->setDynamicRange(-127,127);
    }
  }
  // set dynamic range for layer output tensors
  for (int i = 0; i < network->getNbLayers(); ++i)
  {
    for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j)
    {
      string tName = network->getLayer(i)->getOutput(j)->getName();
      if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end()){
        // Calibrator generated dynamic range for network tensor can be overriden or set using below API
        network->getLayer(i)->getOutput(j)->setDynamicRange(-mPerTensorDynamicRangeMap.at(tName), mPerTensorDynamicRangeMap.at(tName));
      }else{
        logger.log(ILogger::Severity::kERROR,("Could not find "+tName+" from file, use default 127").c_str());
        network->getLayer(i)->getOutput(i)->setDynamicRange(-127,127);
      }
    }
  }
  return true;
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
