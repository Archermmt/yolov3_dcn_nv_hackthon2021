//
// Created by chengjin on 2020-08-24.
//

#include "trt_utils.h"
#include "gelu.h"
#include "kernel/gpu/activate_kernel.h"

#include <cassert>

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

namespace {
    static const char* GELU_PLUGIN_VERSION{"1"};
    static const char* GELU_PLUGIN_NAME{"gelu"};
}

// Static class fields initialization
PluginFieldCollection GELU_Creator::mFC{};
std::vector<PluginField> GELU_Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GELU_Creator);

GELU_Plugin::GELU_Plugin(const std::string name)
    : mLayerName(name)
{
}

GELU_Plugin::GELU_Plugin(const std::string name,const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    readFromBuffer(d,mEleSize);
    readFromBuffer(d,mDataType);

    assert(d == (a + length) && "length of data mismatch");
}

const char* GELU_Plugin::getPluginType() const
{
    return GELU_PLUGIN_NAME;
}

const char* GELU_Plugin::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

int GELU_Plugin::getNbOutputs() const
{
    return 1;
}

Dims GELU_Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);
    mEleSize=1;
    for(int i=0;i<inputs[0].nbDims;i++){
        mEleSize*=inputs[0].d[i];
    }
    return inputs[0];
}

int GELU_Plugin::initialize()
{
    return 0;
}

int GELU_Plugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    if (mDataType == DataType::kFLOAT){
      gelu_forward_gpu(stream,
        reinterpret_cast<const float*>(inputs[0]),
        reinterpret_cast<float*>(outputs[0]),batchSize*mEleSize);
    } else if (mDataType == DataType::kHALF){
      gelu_forward_gpu(stream,
        reinterpret_cast<const __half*>(inputs[0]),
        reinterpret_cast<__half*>(outputs[0]),batchSize*mEleSize);
    }
    return 0;
}

size_t GELU_Plugin::getSerializationSize() const
{
    return 1*sizeof(int)+sizeof(mDataType);
}

void GELU_Plugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mEleSize);
    writeToBuffer(d, mDataType);
    
    assert(d == a + getSerializationSize());
}

void GELU_Plugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(format == PluginFormat::kNCHW);
    mEleSize=1;
    for(int i=0;i<inputs[0].nbDims;i++){
        mEleSize*=inputs[0].d[i];
    }
    mDataType = type;
}

bool GELU_Plugin::supportsFormat(DataType type, PluginFormat format) const
{
    return format == PluginFormat::kNCHW;
}

void GELU_Plugin::terminate() {}

void GELU_Plugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* GELU_Plugin::clone() const
{
    return new GELU_Plugin(mLayerName);
}

void GELU_Plugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* GELU_Plugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

GELU_Creator::GELU_Creator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GELU_Creator::getPluginName() const
{
    return GELU_PLUGIN_NAME;
}

const char* GELU_Creator::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}

const PluginFieldCollection* GELU_Creator::getFieldNames()
{
    return &mFC;
}

IPluginV2* GELU_Creator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    GELU_Plugin* obj = new GELU_Plugin(name);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* GELU_Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    GELU_Plugin* obj = new GELU_Plugin(name, serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void GELU_Creator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* GELU_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
