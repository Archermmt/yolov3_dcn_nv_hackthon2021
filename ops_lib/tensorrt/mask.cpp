//
// Created by chengjin on 2021-02-24.
//

#include "trt_utils.h"
#include "mask.h"
#include "kernel/gpu/mask_kernel.h"

#include <cassert>

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

namespace {
    static const char* MASK_PLUGIN_VERSION{"1"};
    static const char* MASK_PLUGIN_NAME{"mask"};
}

// Static class fields initialization
PluginFieldCollection MASK_Creator::mFC{};
std::vector<PluginField> MASK_Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MASK_Creator);

MASK_Plugin::MASK_Plugin(const std::string name,float value,MaskMode mode)
    : mLayerName(name)
    , mValue(value)
    , mMode(mode)
{
}

MASK_Plugin::MASK_Plugin(const std::string name,const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    readFromBuffer(d,mValue);
    readFromBuffer(d,mMode);
    readFromBuffer(d,mDataSize);
    readFromBuffer(d,mDataType);

    assert(d == (a + length) && "length of data mismatch");
}

const char* MASK_Plugin::getPluginType() const
{
    return MASK_PLUGIN_NAME;
}

const char* MASK_Plugin::getPluginVersion() const
{
    return MASK_PLUGIN_VERSION;
}

int MASK_Plugin::getNbOutputs() const
{
    return 1;
}

Dims MASK_Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);
    int size=1;
    for(int i=0;i<inputs[0].nbDims;i++){
        size*=inputs[0].d[i];
    }
    mDataSize=size;
    return inputs[0];
}

int MASK_Plugin::initialize()
{
    return 0;
}

int MASK_Plugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    if (mDataType == DataType::kFLOAT){
      if(mMode==MaskMode::kNOT_EQUAL){
        mask_ne_forward_gpu(stream,
          reinterpret_cast<const float*>(inputs[0]),
          reinterpret_cast<float*>(outputs[0]),mValue,mDataSize*batchSize);
      }else if(mMode==MaskMode::kGREATER_THAN){
        mask_gt_forward_gpu(stream,
          reinterpret_cast<const float*>(inputs[0]),
          reinterpret_cast<float*>(outputs[0]),mValue,mDataSize*batchSize);
      }else{
        throw std::runtime_error("Unexpected cond type found");
      }
    } else if (mDataType == DataType::kHALF){
      if(mMode==MaskMode::kNOT_EQUAL){
        mask_ne_forward_gpu(stream,
          reinterpret_cast<const __half*>(inputs[0]),
          reinterpret_cast<__half*>(outputs[0]),dlr_float2half(mValue),mDataSize*batchSize);
      }else if(mMode==MaskMode::kGREATER_THAN){
        mask_gt_forward_gpu(stream,
          reinterpret_cast<const __half*>(inputs[0]),
          reinterpret_cast<__half*>(outputs[0]),dlr_float2half(mValue),mDataSize*batchSize);
      }else{
        throw std::runtime_error("Unexpected cond type found");
      }
    }
    return 0;
}

size_t MASK_Plugin::getSerializationSize() const
{
    return sizeof(float)+sizeof(mMode)+sizeof(int)+sizeof(mDataType);
}

void MASK_Plugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mValue);
    writeToBuffer(d, mMode);
    writeToBuffer(d, mDataSize);
    writeToBuffer(d, mDataType);
    
    assert(d == a + getSerializationSize());
}

void MASK_Plugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    // Validate input arguments
    assert(nbInputs == 1);
    int size=1;
    for(int i=0;i<inputs[0].nbDims;i++){
        size*=inputs[0].d[i];
    }
    mDataSize = size;
    mDataType = type;
}

bool MASK_Plugin::supportsFormat(DataType type, PluginFormat format) const
{
    return true;
}

void MASK_Plugin::terminate() {}

void MASK_Plugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* MASK_Plugin::clone() const
{
    return new MASK_Plugin(mLayerName,mValue,mMode);
}

void MASK_Plugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* MASK_Plugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

MASK_Creator::MASK_Creator()
{
    mPluginAttributes.emplace_back(PluginField("value", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 1));
    
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MASK_Creator::getPluginName() const
{
    return MASK_PLUGIN_NAME;
}

const char* MASK_Creator::getPluginVersion() const
{
    return MASK_PLUGIN_VERSION;
}

const PluginFieldCollection* MASK_Creator::getFieldNames()
{
    return &mFC;
}

IPluginV2* MASK_Creator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float value=0;
    MaskMode cond=MaskMode::kNOT_EQUAL;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 4);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "value") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            value = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "mode") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            cond = *(static_cast<const MaskMode*>(fields[i].data));
        }
    }
    MASK_Plugin* obj = new MASK_Plugin(name,value,cond);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* MASK_Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    MASK_Plugin* obj = new MASK_Plugin(name, serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void MASK_Creator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* MASK_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
