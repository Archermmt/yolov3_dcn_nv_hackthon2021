//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_TRT_TRANS_IMAGE_H
#define OPSLIB_TRT_TRANS_IMAGE_H

#include "NvInfer.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

enum class ColorOrder : int
{
  kRGB=0,
  kBGR=1
};

/** transpose image for tensorRT fit to OP.TRANS_IMAGE, 
 * Input ITensor, NbInputs 1, NbOutputs 1
 * Attribute crop height, crop width, offset height, offet width
 * Crop the input tensor along height and width dimension, output shape should be DimCHW{I_channel,crop height,crop width}
 *
 */
class TRANS_IMAGE_Plugin : public IPluginV2
{
public:
    TRANS_IMAGE_Plugin(const std::string name,int image_height,int image_width,PluginFormat layout,ColorOrder color_order);

    TRANS_IMAGE_Plugin(const std::string name,const void* data, size_t length);

    // It doesn't make sense to make crop2d without arguments, so we delete default constructor.
    TRANS_IMAGE_Plugin() = delete;

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
    int mImageHeight,mImageWidth;
    DataType mDataType{DataType::kFLOAT};
    PluginFormat mLayout{PluginFormat::kNCHW};
    ColorOrder mColorOrder{ColorOrder::kRGB};
};

class TRANS_IMAGE_Creator : public IPluginCreator
{
public:
    TRANS_IMAGE_Creator();

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

#endif // OPSLIB_TRT_TRANS_IMAGE_H
