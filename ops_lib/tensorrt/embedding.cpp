//
// Created by chengjin on 2020-06-30.
//

#include "trt_utils.h"
#include "embedding.h"
#include "kernel/gpu/embedding_kernel.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

namespace {
    static const char* EMBEDDING_PLUGIN_VERSION{"1"};
    static const char* EMBEDDING_PLUGIN_NAME{"embedding"};
}

// Static class fields initialization
PluginFieldCollection EMBEDDING_Creator::mFC{};
std::vector<PluginField> EMBEDDING_Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(EMBEDDING_Creator);

EMBEDDING_Plugin::EMBEDDING_Plugin(const std::string name,int emb_num,int emb_dim,const Weights* weights,int nbWeights)
    : mLayerName(name)
    , mEmbeddingNum(emb_num)
    , mEmbeddingDim(emb_dim)
{
    assert(nbWeights==1);
    mWeight = weights[0];
    assert(mWeight.type == DataType::kFLOAT || mWeight.type == DataType::kHALF);
    assert(mWeight.count == mEmbeddingNum*mEmbeddingDim);

    mWeight.values = malloc(mWeight.count * type2size(mWeight.type));
    memcpy(const_cast<void*>(mWeight.values),weights[0].values,mWeight.count*type2size(mWeight.type));
}

EMBEDDING_Plugin::EMBEDDING_Plugin(const std::string name,const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    readFromBuffer(d,mEmbeddingNum);
    readFromBuffer(d,mEmbeddingDim);
    readFromBuffer(d,mIdxNum);
    readFromBuffer(d,mDataType);
    readFromBuffer(d,mWeight.count);
    mWeight.values = nullptr;
    
    deserializeToDevice(d,mDeviceWeight,mWeight.count*type2size(mDataType));
    assert(d == (a + length) && "length of data mismatch");
}

const char* EMBEDDING_Plugin::getPluginType() const
{
    return EMBEDDING_PLUGIN_NAME;
}

const char* EMBEDDING_Plugin::getPluginVersion() const
{
    return EMBEDDING_PLUGIN_VERSION;
}

int EMBEDDING_Plugin::getNbOutputs() const
{
    return 1;
}

Dims EMBEDDING_Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);
    // change the output dim along height and width
    assert(inputs[0].nbDims==1);
    assert(inputs[0].d[0]<mEmbeddingNum);
    mIdxNum=inputs[0].d[0];
    return Dims2{inputs[0].d[0],mEmbeddingDim};
}

int EMBEDDING_Plugin::initialize()
{
    if (mWeight.values)
        convertAndCopyToDevice(mDeviceWeight, mWeight, mDataType);
    return 0;
}

int EMBEDDING_Plugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    if (mDataType == DataType::kFLOAT){
      embedding_forward_gpu(stream,
        reinterpret_cast<const float*>(inputs[0]),
        reinterpret_cast<const float*>(mDeviceWeight),
        reinterpret_cast<float*>(outputs[0]),
        batchSize,mIdxNum,mEmbeddingNum,mEmbeddingDim);
    } else if (mDataType == DataType::kHALF){
      embedding_forward_gpu(stream,
        reinterpret_cast<const __half*>(inputs[0]),
        reinterpret_cast<const __half*>(mDeviceWeight),
        reinterpret_cast<__half*>(outputs[0]),
        batchSize,mIdxNum,mEmbeddingNum,mEmbeddingDim);
    } else if (mDataType == DataType::kINT8){
      std::cout<<"calling int8 datdtype!!"<<std::endl;
    }
    return 0;
}

size_t EMBEDDING_Plugin::getSerializationSize() const
{
    return 3 * sizeof(int)+sizeof(mDataType)+sizeof(mWeight.count) + (mWeight.count) * type2size(mDataType);
}

void EMBEDDING_Plugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mEmbeddingNum);
    writeToBuffer(d, mEmbeddingDim);
    writeToBuffer(d, mIdxNum);
    writeToBuffer(d, mDataType);
    writeToBuffer(d, mWeight.count);
    convertAndCopyToBuffer(d, mWeight, mDataType);
    
    assert(d == a + getSerializationSize());
}

void EMBEDDING_Plugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8));
    
    assert(nbInputs == 1);
    assert(inputs[0].nbDims==1);
    assert(inputs[0].d[0]<mEmbeddingNum);
    mIdxNum=inputs[0].d[0];
    mDataType = type;
}

bool EMBEDDING_Plugin::supportsFormat(DataType type, PluginFormat format) const
{
    //return (type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8);
    return type == DataType::kFLOAT;
}

void EMBEDDING_Plugin::terminate() {
    if (mDeviceWeight)
    {
        cudaFree(mDeviceWeight);
        mDeviceWeight = nullptr;
    }
}

void EMBEDDING_Plugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* EMBEDDING_Plugin::clone() const
{
    return new EMBEDDING_Plugin(mLayerName,mEmbeddingNum,mEmbeddingDim,&mWeight,1);
}

void EMBEDDING_Plugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* EMBEDDING_Plugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

EMBEDDING_Creator::EMBEDDING_Creator()
{
    mPluginAttributes.emplace_back(PluginField("emb_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("emb_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weight", nullptr, PluginFieldType::kFLOAT32, 1));
    
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EMBEDDING_Creator::getPluginName() const
{
    return EMBEDDING_PLUGIN_NAME;
}

const char* EMBEDDING_Creator::getPluginVersion() const
{
    return EMBEDDING_PLUGIN_VERSION;
}

const PluginFieldCollection* EMBEDDING_Creator::getFieldNames()
{
    return &mFC;
}

IPluginV2* EMBEDDING_Creator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int emb_num=0;
    int emb_dim=0;
    std::vector<float> weightValues;
    const PluginField* fields = fc->fields;
    assert(fc->nbFields == 1); 
    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (strcmp(fields[i].name,"weight")==0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            weightValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                weightValues.push_back(*w);
                w++;
            }
        } else if (strcmp(fields[i].name, "emb_num") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            emb_num = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "emb_dim") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            emb_dim = *(static_cast<const float*>(fields[i].data));
        }
    }
    Weights weight{DataType::kFLOAT, weightValues.data(), (int64_t) weightValues.size()};
    
    EMBEDDING_Plugin* obj = new EMBEDDING_Plugin(name,emb_num,emb_dim,&weight,1);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* EMBEDDING_Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    EMBEDDING_Plugin* obj = new EMBEDDING_Plugin(name,serialData,serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void EMBEDDING_Creator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* EMBEDDING_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
