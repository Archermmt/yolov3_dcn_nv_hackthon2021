//
// Created by chengjin on 2020-06-02.
//

#include "trt_utils.h"
#include "group_norm.h"
#include "kernel/gpu/normalize_kernel.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

namespace {
    static const char* GROUP_NORM_PLUGIN_VERSION{"1"};
    static const char* GROUP_NORM_PLUGIN_NAME{"group_norm"};
}

// Static class fields initialization
PluginFieldCollection GROUP_NORM_Creator::mFC{};
std::vector<PluginField> GROUP_NORM_Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GROUP_NORM_Creator);

GROUP_NORM_Plugin::GROUP_NORM_Plugin(const std::string name,float eps,int num_group,const Weights* weights,int nbWeights)
    : mLayerName(name)
    , mEps(eps)
    , mNumGroup(num_group)
{
    assert(nbWeights==2);

    mGamma = weights[0];
    assert(mGamma.type == DataType::kFLOAT || mGamma.type == DataType::kHALF);
    mGamma.values = malloc(mGamma.count * type2size(mGamma.type));
    memcpy(const_cast<void*>(mGamma.values),weights[0].values,mGamma.count*type2size(mGamma.type));

    mBeta = weights[1];
    assert(mBeta.type == DataType::kFLOAT || mBeta.type == DataType::kHALF);
    mBeta.values = malloc(mBeta.count * type2size(mBeta.type));
    memcpy(const_cast<void*>(mBeta.values),weights[1].values,mBeta.count*type2size(mBeta.type));
}

GROUP_NORM_Plugin::GROUP_NORM_Plugin(const std::string name,const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    readFromBuffer(d, mEps);
    readFromBuffer(d, mNumGroup);
    readFromBuffer(d, mInputChannel);
    readFromBuffer(d, mInputHeight);
    readFromBuffer(d, mInputWidth);
    readFromBuffer(d, mDataType);

    readFromBuffer(d,mGamma.count);
    mGamma.values = nullptr;
    deserializeToDevice(d,mDeviceGamma,mGamma.count*type2size(mDataType));
    
    readFromBuffer(d,mBeta.count);
    mBeta.values = nullptr;
    deserializeToDevice(d,mDeviceBeta,mBeta.count*type2size(mDataType));
    
    assert(d == (a + length) && "length of data mismatch");
}

const char* GROUP_NORM_Plugin::getPluginType() const
{
    return GROUP_NORM_PLUGIN_NAME;
}

const char* GROUP_NORM_Plugin::getPluginVersion() const
{
    return GROUP_NORM_PLUGIN_VERSION;
}

int GROUP_NORM_Plugin::getNbOutputs() const
{
    return 1;
}

Dims GROUP_NORM_Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);
    mInputChannel=inputs->d[0];
    mInputHeight=inputs->d[1];
    mInputWidth=inputs->d[2];
    return DimsCHW{mInputChannel,mInputHeight,mInputWidth};
}

int GROUP_NORM_Plugin::initialize()
{
    if (mGamma.values)
        convertAndCopyToDevice(mDeviceGamma,mGamma,mDataType);
    if (mBeta.values)
        convertAndCopyToDevice(mDeviceBeta,mBeta,mDataType);
    return 0;
}

int GROUP_NORM_Plugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    CHECK(cudaMemset(workspace,0,getWorkspaceSize(batchSize)));
    size_t offset=0;
    void* deviceBuffer=workspace+offset;
    offset+=batchSize*mInputChannel*mInputHeight*mInputWidth*sizeof(mDataType);
    void* deviceMean=workspace+offset;
    offset+=batchSize*mNumGroup*sizeof(mDataType);
    void* deviceVar=workspace+offset;
    offset+=batchSize*mNumGroup*sizeof(mDataType);
    if (mDataType == DataType::kFLOAT){
      groupnorm_forward_gpu(stream,
        reinterpret_cast<const float*>(inputs[0]),
        reinterpret_cast<const float*>(mDeviceGamma),
        reinterpret_cast<const float*>(mDeviceBeta),
        reinterpret_cast<float*>(deviceBuffer),
        reinterpret_cast<float*>(deviceMean),
        reinterpret_cast<float*>(deviceVar),
        reinterpret_cast<float*>(outputs[0]),
        mNumGroup,batchSize,mInputChannel,mInputHeight,mInputWidth,mEps);
    } else if (mDataType == DataType::kHALF){
      groupnorm_forward_gpu(stream,
        reinterpret_cast<const __half*>(inputs[0]),
        reinterpret_cast<const __half*>(mDeviceGamma),
        reinterpret_cast<const __half*>(mDeviceBeta),
        reinterpret_cast<__half*>(deviceBuffer),
        reinterpret_cast<__half*>(deviceMean),
        reinterpret_cast<__half*>(deviceVar),
        reinterpret_cast<__half*>(outputs[0]),
        mNumGroup,batchSize,mInputChannel,mInputHeight,mInputWidth,dlr_float2half(mEps));
    }
    CHECK(cudaStreamSynchronize(stream));
    return 0;
}

size_t GROUP_NORM_Plugin::getSerializationSize() const
{
    return 1*sizeof(float)+4*sizeof(int)+sizeof(mDataType)+ \
    sizeof(mGamma.count)+(mGamma.count)*type2size(mDataType)+ \
    sizeof(mBeta.count)+(mBeta.count)*type2size(mDataType);
}

void GROUP_NORM_Plugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mEps);
    writeToBuffer(d, mNumGroup);
    writeToBuffer(d, mInputChannel);
    writeToBuffer(d, mInputHeight);
    writeToBuffer(d, mInputWidth);
    writeToBuffer(d, mDataType);
    
    writeToBuffer(d, mGamma.count);
    convertAndCopyToBuffer(d, mGamma, mDataType);
    writeToBuffer(d, mBeta.count);
    convertAndCopyToBuffer(d, mBeta, mDataType);

    assert(d == a + getSerializationSize());
}

void GROUP_NORM_Plugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(format == PluginFormat::kNCHW);
    mInputChannel=inputs->d[0];
    mInputHeight=inputs->d[1];
    mInputWidth=inputs->d[2];
    mDataType = type;
}

bool GROUP_NORM_Plugin::supportsFormat(DataType type, PluginFormat format) const
{
    return format == PluginFormat::kNCHW;
}

void GROUP_NORM_Plugin::terminate() {
    if (mDeviceGamma)
    {
        cudaFree(mDeviceGamma);
        mDeviceGamma = nullptr;
    }
    if (mDeviceBeta)
    {
        cudaFree(mDeviceBeta);
        mDeviceBeta = nullptr;
    }
}

void GROUP_NORM_Plugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* GROUP_NORM_Plugin::clone() const
{
    Weights weights[2]={mGamma,mBeta};
    return new GROUP_NORM_Plugin(mLayerName,mEps,mNumGroup,weights,2);
}

void GROUP_NORM_Plugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* GROUP_NORM_Plugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

GROUP_NORM_Creator::GROUP_NORM_Creator()
{
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_group", nullptr, PluginFieldType::kINT32, 1));
    
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GROUP_NORM_Creator::getPluginName() const
{
    return GROUP_NORM_PLUGIN_NAME;
}

const char* GROUP_NORM_Creator::getPluginVersion() const
{
    return GROUP_NORM_PLUGIN_VERSION;
}

const PluginFieldCollection* GROUP_NORM_Creator::getFieldNames()
{
    return &mFC;
}

IPluginV2* GROUP_NORM_Creator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float eps=0;
    int num_group=0;
    std::vector<float> gammaValues;
    std::vector<float> betaValues;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 2);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "eps") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            eps = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "num_group") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            num_group = *(static_cast<const float*>(fields[i].data));
        }
    }
    Weights gamma{DataType::kFLOAT, gammaValues.data(), (int64_t) gammaValues.size()};
    Weights beta{DataType::kFLOAT, betaValues.data(), (int64_t) betaValues.size()};

    Weights weights[2]={gamma,beta};
    GROUP_NORM_Plugin* obj = new GROUP_NORM_Plugin(name,eps,num_group,weights,2);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* GROUP_NORM_Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    GROUP_NORM_Plugin* obj = new GROUP_NORM_Plugin(name, serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void GROUP_NORM_Creator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* GROUP_NORM_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
