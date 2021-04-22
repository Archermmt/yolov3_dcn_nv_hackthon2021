//
// Created by chengjin on 2020-06-02.
//

#include "trt_utils.h"
#include "leaky_relu.h"
#include "kernel/gpu/activate_kernel.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

namespace {
    static const char* LEAKY_RELU_PLUGIN_VERSION{"1"};
    static const char* LEAKY_RELU_PLUGIN_NAME{"leaky_relu"};
}

// Static class fields initialization
PluginFieldCollection LEAKY_RELU_Creator::mFC{};
std::vector<PluginField> LEAKY_RELU_Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(LEAKY_RELU_Creator);

LEAKY_RELU_Plugin::LEAKY_RELU_Plugin(const std::string name,const Weights* weights,int nbWeights)
    : mLayerName(name)
{
    assert(nbWeights==1);
    mGammaWeights = weights[0];
    assert(mGammaWeights.type == DataType::kFLOAT || mGammaWeights.type == DataType::kHALF);
    
    mGammaWeights.values = malloc(mGammaWeights.count * type2size(mGammaWeights.type));
    memcpy(const_cast<void*>(mGammaWeights.values),weights[0].values,mGammaWeights.count*type2size(mGammaWeights.type));
}

LEAKY_RELU_Plugin::LEAKY_RELU_Plugin(const std::string name,const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    readFromBuffer(d,mInputChannel);
    readFromBuffer(d,mInputHeight);
    readFromBuffer(d,mInputWidth);
    readFromBuffer(d,mDataType);
    readFromBuffer(d,mGammaWeights.count);
    mGammaWeights.values = nullptr;
    
    deserializeToDevice(d,mDeviceGamma,mGammaWeights.count*type2size(mDataType));
    assert(d == (a + length) && "length of data mismatch");
}

const char* LEAKY_RELU_Plugin::getPluginType() const
{
    return LEAKY_RELU_PLUGIN_NAME;
}

const char* LEAKY_RELU_Plugin::getPluginVersion() const
{
    return LEAKY_RELU_PLUGIN_VERSION;
}

int LEAKY_RELU_Plugin::getNbOutputs() const
{
    return 1;
}

Dims LEAKY_RELU_Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);
    // change the output dim along height and width
    mInputChannel=inputs->d[0];
    mInputHeight=inputs->d[1];
    mInputWidth=inputs->d[2];
    return DimsCHW{mInputChannel,mInputHeight,mInputWidth};
}

int LEAKY_RELU_Plugin::initialize()
{
    if (mGammaWeights.values)
        convertAndCopyToDevice(mDeviceGamma, mGammaWeights, mDataType);
    return 0;
}

int LEAKY_RELU_Plugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    if (mDataType == DataType::kFLOAT){
      leaky_relu_forward_gpu(stream,
        reinterpret_cast<const float*>(inputs[0]),
        reinterpret_cast<const float*>(mDeviceGamma),
        reinterpret_cast<float*>(outputs[0]),
        batchSize,mInputChannel,mInputHeight,mInputWidth);
    } else if (mDataType == DataType::kHALF){
      leaky_relu_forward_gpu(stream,
        reinterpret_cast<const __half*>(inputs[0]),
        reinterpret_cast<const __half*>(mDeviceGamma),
        reinterpret_cast<__half*>(outputs[0]),
        batchSize,mInputChannel,mInputHeight,mInputWidth);
    } else if (mDataType == DataType::kINT8){
      std::cout<<"calling int8 datdtype!!"<<std::endl;
    }
    return 0;
}

size_t LEAKY_RELU_Plugin::getSerializationSize() const
{
    return 3 * sizeof(int)+sizeof(mDataType)+sizeof(mGammaWeights.count) + (mGammaWeights.count) * type2size(mDataType);
}

void LEAKY_RELU_Plugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mInputChannel);
    writeToBuffer(d, mInputHeight);
    writeToBuffer(d, mInputWidth);
    writeToBuffer(d, mDataType);
    writeToBuffer(d, mGammaWeights.count);
    convertAndCopyToBuffer(d, mGammaWeights, mDataType);
    
    assert(d == a + getSerializationSize());
}

void LEAKY_RELU_Plugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8) && format == PluginFormat::kNCHW);
    mInputChannel=inputs->d[0];
    mInputHeight=inputs->d[1];
    mInputWidth=inputs->d[2];
    mDataType = type;
}

bool LEAKY_RELU_Plugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8) && format == PluginFormat::kNCHW;
}

void LEAKY_RELU_Plugin::terminate() {
    if (mDeviceGamma)
    {
        cudaFree(mDeviceGamma);
        mDeviceGamma = nullptr;
    }
}

void LEAKY_RELU_Plugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* LEAKY_RELU_Plugin::clone() const
{
    return new LEAKY_RELU_Plugin(mLayerName,&mGammaWeights,1);
}

void LEAKY_RELU_Plugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* LEAKY_RELU_Plugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

LEAKY_RELU_Creator::LEAKY_RELU_Creator()
{
    mPluginAttributes.emplace_back(PluginField("gamma", nullptr, PluginFieldType::kFLOAT32, 1));
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LEAKY_RELU_Creator::getPluginName() const
{
    return LEAKY_RELU_PLUGIN_NAME;
}

const char* LEAKY_RELU_Creator::getPluginVersion() const
{
    return LEAKY_RELU_PLUGIN_VERSION;
}

const PluginFieldCollection* LEAKY_RELU_Creator::getFieldNames()
{
    return &mFC;
}

IPluginV2* LEAKY_RELU_Creator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    std::vector<float> gammaValues;
    const PluginField* fields = fc->fields;
    assert(fc->nbFields == 1); 
    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (strcmp(fields[i].name,"gamma")==0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            gammaValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                gammaValues.push_back(*w);
                w++;
            }
        }
    }
    Weights gammaWeigths{DataType::kFLOAT, gammaValues.data(), (int64_t) gammaValues.size()};
    
    LEAKY_RELU_Plugin* obj = new LEAKY_RELU_Plugin(name,&gammaWeigths,1);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* LEAKY_RELU_Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    LEAKY_RELU_Plugin* obj = new LEAKY_RELU_Plugin(name,serialData,serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void LEAKY_RELU_Creator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* LEAKY_RELU_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
