//
// Auto Generated by DLRouter on 2021-03-15 08:57:28.079692
// 

#ifndef OPSLIB_TRT_NONLINEAR_PRED_BOX_H
#define OPSLIB_TRT_NONLINEAR_PRED_BOX_H

#include "NvInfer.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

class NONLINEAR_PRED_BOX_Plugin : public IPluginV2
{
public:
  //initialize functions
  NONLINEAR_PRED_BOX_Plugin(const std::string name,int p_image_h,int p_image_w);

  NONLINEAR_PRED_BOX_Plugin(const std::string name,const void* data, size_t length);

  NONLINEAR_PRED_BOX_Plugin() = delete;

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
  int image_h;
  int image_w;
  //def the dims
  int box_num;
  int box_dim;
};

class NONLINEAR_PRED_BOX_Creator : public IPluginCreator
{
public:
  NONLINEAR_PRED_BOX_Creator();

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

#endif // OPSLIB_TRT_NONLINEAR_PRED_BOX_H
