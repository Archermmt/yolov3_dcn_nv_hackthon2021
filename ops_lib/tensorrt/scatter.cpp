//
// Created by chengjin on 2020-06-30.
//

#include <exception>
#include <cassert>
#include "trt_utils.h"
#include "scatter.h"
#include "kernel/gpu/scatter_kernel.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

namespace {
    static const char* SCATTER_PLUGIN_VERSION{"1"};
    static const char* SCATTER_PLUGIN_NAME{"scatter"};
}

// Static class fields initialization
PluginFieldCollection SCATTER_Creator::mFC{};
std::vector<PluginField> SCATTER_Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SCATTER_Creator);

SCATTER_Plugin::SCATTER_Plugin(const std::string name,float value,ScatterCond cond)
    : mLayerName(name)
    , mValue(value)
    , mCond(cond)
{
}

SCATTER_Plugin::SCATTER_Plugin(const std::string name,const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    readFromBuffer(d,mValue);
    readFromBuffer(d,mCond);
    readFromBuffer(d,mDataSize);
    readFromBuffer(d,mDataType);

    assert(d == (a + length) && "length of data mismatch");
}

const char* SCATTER_Plugin::getPluginType() const
{
    return SCATTER_PLUGIN_NAME;
}

const char* SCATTER_Plugin::getPluginVersion() const
{
    return SCATTER_PLUGIN_VERSION;
}

int SCATTER_Plugin::getNbOutputs() const
{
    return 1;
}

Dims SCATTER_Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 2);
    assert(index == 0);
    //input1 shoula be shape shape as input 0 or input0.expand_dims()
    assert(inputs[0].nbDims==inputs[1].nbDims || inputs[0].nbDims==(inputs[1].nbDims-1));
    int size=1;
    for(int i=0;i<inputs[0].nbDims;i++){
        assert(inputs[0].d[i]==inputs[1].d[i]);
        size*=inputs[0].d[i];
    }
    if(inputs[0].nbDims==(inputs[1].nbDims-1)){
      assert(inputs[1].d[inputs[0].nbDims]==1);
    }
    mDataSize=size;
    return inputs[0];
}

int SCATTER_Plugin::initialize()
{
    return 0;
}

int SCATTER_Plugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    if (mDataType == DataType::kFLOAT){
      if(mCond==ScatterCond::kNOT_EQUAL){
        scatter_ne_forward_gpu(stream,
          reinterpret_cast<const float*>(inputs[0]),
          reinterpret_cast<const float*>(inputs[1]),
          reinterpret_cast<float*>(outputs[0]),mValue,mDataSize*batchSize);
      }else{
        throw std::runtime_error("Unexpected cond type found");
      }
    } else if (mDataType == DataType::kHALF){
      if(mCond==ScatterCond::kNOT_EQUAL){
        scatter_ne_forward_gpu(stream,
          reinterpret_cast<const __half*>(inputs[0]),
          reinterpret_cast<const __half*>(inputs[1]),
          reinterpret_cast<__half*>(outputs[0]),dlr_float2half(mValue),mDataSize*batchSize);
      }else{
        throw std::runtime_error("Unexpected cond type found");
      }
    }
    return 0;
}

size_t SCATTER_Plugin::getSerializationSize() const
{
    return sizeof(float)+sizeof(mCond)+sizeof(int)+sizeof(mDataType);
}

void SCATTER_Plugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mValue);
    writeToBuffer(d, mCond);
    writeToBuffer(d, mDataSize);
    writeToBuffer(d, mDataType);
    
    assert(d == a + getSerializationSize());
}

void SCATTER_Plugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    // Validate input arguments
    assert(nbInputs == 2);
    assert(inputs[0].nbDims==inputs[1].nbDims || inputs[0].nbDims==(inputs[1].nbDims-1));
    int size=1;
    for(int i=0;i<inputs[0].nbDims;i++){
        assert(inputs[0].d[i]==inputs[1].d[i]);
        size*=inputs[0].d[i];
    }
    if(inputs[0].nbDims==(inputs[1].nbDims-1)){
      assert(inputs[1].d[inputs[0].nbDims]==1);
    }
    mDataSize = size;
    mDataType = type;
}

bool SCATTER_Plugin::supportsFormat(DataType type, PluginFormat format) const
{
    return true;
}

void SCATTER_Plugin::terminate() {}

void SCATTER_Plugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* SCATTER_Plugin::clone() const
{
    return new SCATTER_Plugin(mLayerName,mValue,mCond);
}

void SCATTER_Plugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* SCATTER_Plugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

SCATTER_Creator::SCATTER_Creator()
{
    mPluginAttributes.emplace_back(PluginField("value", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("cond", nullptr, PluginFieldType::kINT32, 1));
    
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SCATTER_Creator::getPluginName() const
{
    return SCATTER_PLUGIN_NAME;
}

const char* SCATTER_Creator::getPluginVersion() const
{
    return SCATTER_PLUGIN_VERSION;
}

const PluginFieldCollection* SCATTER_Creator::getFieldNames()
{
    return &mFC;
}

IPluginV2* SCATTER_Creator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float value=0;
    ScatterCond cond=ScatterCond::kNOT_EQUAL;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 4);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "value") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            value = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "cond") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            cond = *(static_cast<const ScatterCond*>(fields[i].data));
        }
    }
    SCATTER_Plugin* obj = new SCATTER_Plugin(name,value,cond);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* SCATTER_Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    SCATTER_Plugin* obj = new SCATTER_Plugin(name, serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void SCATTER_Creator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* SCATTER_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
