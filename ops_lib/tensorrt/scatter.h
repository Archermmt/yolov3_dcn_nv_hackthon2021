//
// Created by chengjin on 2020-06-30.
//

#ifndef OPSLIB_TRT_SCATTER_H
#define OPSLIB_TRT_SCATTER_H

#include "NvInfer.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

enum class ScatterCond : int
{
  kNOT_EQUAL=0,
  kBGR=1
};

class SCATTER_Plugin : public IPluginV2
{
public:
    SCATTER_Plugin(const std::string name,float value,ScatterCond cond);

    SCATTER_Plugin(const std::string name,const void* data, size_t length);

    // It doesn't make sense to make crop2d without arguments, so we delete default constructor.
    SCATTER_Plugin() = delete;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    const std::string mLayerName;
    std::string mNamespace;
    DataType mDataType{DataType::kFLOAT};
    float mValue;
    ScatterCond mCond{ScatterCond::kNOT_EQUAL};
    int mDataSize;
};

class SCATTER_Creator : public IPluginCreator
{
public:
    SCATTER_Creator();

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

#endif // OPSLIB_TRT_SCATTER_H
