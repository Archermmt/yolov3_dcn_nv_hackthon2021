//
// Auto Generated by DLRouter on 2021-03-19 08:09:34.938741
// 

#ifndef OPSLIB_TRT_BILINEAR_UPSAMPLE_H
#define OPSLIB_TRT_BILINEAR_UPSAMPLE_H

#include "NvInfer.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

class BILINEAR_UPSAMPLE_Plugin : public IPluginV2
{
public:
  //initialize functions
  BILINEAR_UPSAMPLE_Plugin(const std::string name,int p_out_h,int p_out_w,bool p_align_corners);

  BILINEAR_UPSAMPLE_Plugin(const std::string name,const void* data, size_t length);

  BILINEAR_UPSAMPLE_Plugin() = delete;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void serialize(void* buffer) const override;

  size_t getSerializationSize() const override;

  int initialize() override;

  void terminate() override;

  void destroy() override;

  //runtime functions
  int getNbOutputs() const override;

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

  size_t getWorkspaceSize(int) const override;

  int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

  //config functions
  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

  bool supportsFormat(DataType type, PluginFormat format) const override;

  //describe functions
  nvinfer1::IPluginV2* clone() const override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

private:
  const std::string mLayerName;
  std::string mNamespace;
  DataType mDataType{DataType::kFLOAT};
  //def the attributes
  int out_h;
  int out_w;
  bool align_corners;
  //def the dims
  int channel;
  int height;
  int width;
};

class BILINEAR_UPSAMPLE_Creator : public IPluginCreator
{
public:
  BILINEAR_UPSAMPLE_Creator();

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const PluginFieldCollection* getFieldNames() override;

  IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

  IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mNamespace;
};

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif // OPSLIB_TRT_BILINEAR_UPSAMPLE_H
