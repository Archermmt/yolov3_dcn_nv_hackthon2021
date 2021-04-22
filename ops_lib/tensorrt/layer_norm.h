//
// Created by chengjin on 2020-07-01.
//

#ifndef OPSLIB_TRT_LAYER_NORM_H
#define OPSLIB_TRT_LAYER_NORM_H

#include "NvInfer.h"
#include <string>

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

/** layer norm for tensorRT fit to OP.LAYER_NORM, 
 * Input ITensor, NbInputs 1, NbOutputs 1
 * Attribute eps,layer_pos
 */

class LAYER_NORM_Plugin : public IPluginV2
{
public:
    LAYER_NORM_Plugin(const std::string name,float eps,int layer_pos,const Weights* weights,int nbWeights);

    LAYER_NORM_Plugin(const std::string name,const void* data, size_t length);

    // It doesn't make sense to make crop2d without arguments, so we delete default constructor.
    LAYER_NORM_Plugin() = delete;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        size_t size_buffer       = maxBatchSize*mLayerLen*mLayerDim*sizeof(mDataType);
        size_t size_mean         = maxBatchSize*mLayerLen*sizeof(mDataType);
        size_t size_var          = maxBatchSize*mLayerLen*sizeof(mDataType);
        size_t total_size=size_buffer+size_mean+size_var;
        return total_size;
    }

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
    float mEps{1e-5};
    int mLayerPos{0};
    int mLayerLen,mLayerDim;
    Weights mGamma;
    Weights mBeta;
    void* mDeviceGamma{nullptr};
    void* mDeviceBeta{nullptr};
};

class LAYER_NORM_Creator : public IPluginCreator
{
public:
    LAYER_NORM_Creator();

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

#endif // OPSLIB_TRT_LAYER_NORM_H
