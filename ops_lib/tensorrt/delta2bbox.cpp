//
// Created by chengjin on 2020-06-02.
//

#include "trt_utils.h"
#include "delta2bbox.h"
#include "kernel/gpu/detect_kernel.h"

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

namespace {
    static const char* DELTA2BBOX_PLUGIN_VERSION{"1"};
    static const char* DELTA2BBOX_PLUGIN_NAME{"delta2bbox"};
}

// Static class fields initialization
PluginFieldCollection DELTA2BBOX_Creator::mFC{};
std::vector<PluginField> DELTA2BBOX_Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DELTA2BBOX_Creator);

DELTA2BBOX_Plugin::DELTA2BBOX_Plugin(const std::string name,int max_height,int max_width,float max_ratio)
    : mLayerName(name)
    , mMaxHeight(max_height)
    , mMaxWidth(max_width)
    , mMaxRatio(max_ratio)
{
}

DELTA2BBOX_Plugin::DELTA2BBOX_Plugin(const std::string name,const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    readFromBuffer(d,mMaxHeight);
    readFromBuffer(d,mMaxWidth);
    readFromBuffer(d,mBoxesNum);
    readFromBuffer(d,mMaxRatio);
    readFromBuffer(d,mDataType);

    assert(d == (a + length) && "length of data mismatch");
}

const char* DELTA2BBOX_Plugin::getPluginType() const
{
    return DELTA2BBOX_PLUGIN_NAME;
}

const char* DELTA2BBOX_Plugin::getPluginVersion() const
{
    return DELTA2BBOX_PLUGIN_VERSION;
}

int DELTA2BBOX_Plugin::getNbOutputs() const
{
    return 1;
}

Dims DELTA2BBOX_Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 3);
    mBoxesNum=inputs[0].d[0];
    //inputs: delta [row,4], rois [row,4], score [row,1]
    for(int i=0;i<2;i++){
        assert(inputs[i].nbDims==2);
        assert(inputs[i].d[0]==mBoxesNum);
        assert(inputs[i].d[1]==4);
    }
    assert(inputs[2].nbDims==2);
    assert(inputs[2].d[0]==mBoxesNum);
    assert(inputs[2].d[1]==1);
    return Dims2{mBoxesNum,5};
}

int DELTA2BBOX_Plugin::initialize()
{
    return 0;
}

int DELTA2BBOX_Plugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    if (mDataType == DataType::kFLOAT){
        delta2bbox_forward_gpu(stream,
            reinterpret_cast<const float*>(inputs[0]),
            reinterpret_cast<const float*>(inputs[1]),
            reinterpret_cast<const float*>(inputs[2]),
            reinterpret_cast<float*>(outputs[0]),
            batchSize*mBoxesNum,mMaxHeight,mMaxWidth,mMaxRatio);
    } else if (mDataType == DataType::kHALF){
        delta2bbox_forward_gpu(stream,
            reinterpret_cast<const __half*>(inputs[0]),
            reinterpret_cast<const __half*>(inputs[1]),
            reinterpret_cast<const __half*>(inputs[2]),
            reinterpret_cast<__half*>(outputs[0]),
            batchSize*mBoxesNum,mMaxHeight,mMaxWidth,dlr_float2half(mMaxRatio));
    }
    return 0;
}

size_t DELTA2BBOX_Plugin::getSerializationSize() const
{
    return 3*sizeof(int)+1*sizeof(float)+sizeof(mDataType);
}

void DELTA2BBOX_Plugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mMaxHeight);
    writeToBuffer(d, mMaxWidth);
    writeToBuffer(d, mBoxesNum);
    writeToBuffer(d, mMaxRatio);
    writeToBuffer(d, mDataType);
    
    assert(d == a + getSerializationSize());
}

void DELTA2BBOX_Plugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbInputs == 3);
    mBoxesNum=inputs[0].d[0];
    //inputs: delta [row,4], rois [row,4], score [row,1]
    for(int i=0;i<2;i++){
        assert(inputs[i].nbDims==2);
        assert(inputs[i].d[0]==mBoxesNum);
        assert(inputs[i].d[1]==4);
    }
    assert(inputs[2].nbDims==2);
    assert(inputs[2].d[0]==mBoxesNum);
    assert(inputs[2].d[1]==1);
    mDataType = type;
}

bool DELTA2BBOX_Plugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT || type == DataType::kHALF);
}

void DELTA2BBOX_Plugin::terminate() {}

void DELTA2BBOX_Plugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* DELTA2BBOX_Plugin::clone() const
{
    return new DELTA2BBOX_Plugin(mLayerName,mMaxHeight,mMaxWidth,mMaxRatio);
}

void DELTA2BBOX_Plugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* DELTA2BBOX_Plugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

DELTA2BBOX_Creator::DELTA2BBOX_Creator()
{
    mPluginAttributes.emplace_back(PluginField("max_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_ratio", nullptr, PluginFieldType::kFLOAT32, 1));
    
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DELTA2BBOX_Creator::getPluginName() const
{
    return DELTA2BBOX_PLUGIN_NAME;
}

const char* DELTA2BBOX_Creator::getPluginVersion() const
{
    return DELTA2BBOX_PLUGIN_VERSION;
}

const PluginFieldCollection* DELTA2BBOX_Creator::getFieldNames()
{
    return &mFC;
}

IPluginV2* DELTA2BBOX_Creator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int max_height=0;
    int max_wdith=0;
    float max_ratio=0.0;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 3);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "max_height") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            max_height = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "max_wdith") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            max_wdith = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "max_ratio") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            max_ratio = *(static_cast<const float*>(fields[i].data));
        }
    }
    DELTA2BBOX_Plugin* obj = new DELTA2BBOX_Plugin(name,max_height,max_wdith,max_ratio);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* DELTA2BBOX_Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    DELTA2BBOX_Plugin* obj = new DELTA2BBOX_Plugin(name, serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void DELTA2BBOX_Creator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* DELTA2BBOX_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
