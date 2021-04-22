//
// Created by chengjin on 2020-06-02.
//

#include "trt_utils.h"
#include "trans_image.h"
#include "kernel/gpu/trans_image_kernel.h"

#include <cassert>

using namespace nvinfer1;

namespace quake {
namespace framework {
namespace ops_lib {

namespace {
    static const char* TRANS_IMAGE_PLUGIN_VERSION{"1"};
    static const char* TRANS_IMAGE_PLUGIN_NAME{"trans_image"};
}

// Static class fields initialization
PluginFieldCollection TRANS_IMAGE_Creator::mFC{};
std::vector<PluginField> TRANS_IMAGE_Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(TRANS_IMAGE_Creator);

TRANS_IMAGE_Plugin::TRANS_IMAGE_Plugin(const std::string name,int image_height,int image_width,PluginFormat layout,ColorOrder color_order)
    : mLayerName(name)
    , mImageHeight(image_height)
    , mImageWidth(image_width)
    , mLayout(layout)
    , mColorOrder(color_order)
{
}

TRANS_IMAGE_Plugin::TRANS_IMAGE_Plugin(const std::string name,const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    readFromBuffer(d,mImageHeight);
    readFromBuffer(d,mImageWidth);
    readFromBuffer(d,mDataType);
    readFromBuffer(d,mLayout);
    readFromBuffer(d,mColorOrder);

    assert(d == (a + length) && "length of data mismatch");
}

const char* TRANS_IMAGE_Plugin::getPluginType() const
{
    return TRANS_IMAGE_PLUGIN_NAME;
}

const char* TRANS_IMAGE_Plugin::getPluginVersion() const
{
    return TRANS_IMAGE_PLUGIN_VERSION;
}

int TRANS_IMAGE_Plugin::getNbOutputs() const
{
    return 1;
}

Dims TRANS_IMAGE_Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);
    // change the output dim along height and width
    if(mDataType==DataType::kINT8){
      // input is DimsCHW tensor, [height,width,tensor]
      assert(inputs[0].nbDims==3);
      assert(inputs[0].d[0]==mImageHeight);
      assert(inputs[0].d[1]==mImageWidth);
      assert(inputs[0].d[2]==3);
    }else{
      // input is Dims2 tensor, [1,datasize] or Dims1 tensor [datasize]
      assert(inputs[0].nbDims==2 || inputs[0].nbDims==1);
      if(inputs[0].nbDims==2){
        assert(inputs[0].d[0]==1);
        assert(inputs[0].d[1]==int(3*mImageHeight*mImageWidth/sizeof(float)));
      }else if(inputs[0].nbDims==1){
        assert(inputs[0].d[0]==int(3*mImageHeight*mImageWidth/sizeof(float)));
      }
    }
    return DimsCHW{3,mImageHeight,mImageWidth};
}

int TRANS_IMAGE_Plugin::initialize()
{
    return 0;
}

int TRANS_IMAGE_Plugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    if (mLayout==PluginFormat::kNCHW){
        if (mDataType == DataType::kFLOAT){
            if(mColorOrder==ColorOrder::kRGB){
                trans_image_NCHW_RGB_gpu(stream,
                    reinterpret_cast<const char*>(inputs[0]),
                    reinterpret_cast<float*>(outputs[0]),
                    batchSize,3,mImageHeight,mImageWidth);
            }else{
                trans_image_NCHW_BGR_gpu(stream,
                    reinterpret_cast<const char*>(inputs[0]),
                    reinterpret_cast<float*>(outputs[0]),
                    batchSize,3,mImageHeight,mImageWidth);
            }
        } else if (mDataType == DataType::kHALF){
            if(mColorOrder==ColorOrder::kRGB){
                trans_image_NCHW_RGB_gpu(stream,
                    reinterpret_cast<const char*>(inputs[0]),
                    reinterpret_cast<__half*>(outputs[0]),
                    batchSize,3,mImageHeight,mImageWidth);
            }else{
                trans_image_NCHW_BGR_gpu(stream,
                    reinterpret_cast<const char*>(inputs[0]),
                    reinterpret_cast<__half*>(outputs[0]),
                    batchSize,3,mImageHeight,mImageWidth);
            }
        }
    } else {
        throw std::runtime_error("Other layout is not supported");
    }
    return 0;
}

size_t TRANS_IMAGE_Plugin::getSerializationSize() const
{
    return 2*sizeof(int)+sizeof(mDataType)+sizeof(mLayout)+sizeof(mColorOrder);
}

void TRANS_IMAGE_Plugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mImageHeight);
    writeToBuffer(d, mImageWidth);
    writeToBuffer(d, mDataType);
    writeToBuffer(d, mLayout);
    writeToBuffer(d, mColorOrder);
    
    assert(d == a + getSerializationSize());
}

void TRANS_IMAGE_Plugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert((type==DataType::kFLOAT || type==DataType::kINT8) && format == PluginFormat::kNCHW);
    mDataType = type;
}

bool TRANS_IMAGE_Plugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type==DataType::kFLOAT || type==DataType::kINT8) && format == PluginFormat::kNCHW;
}

void TRANS_IMAGE_Plugin::terminate() {}

void TRANS_IMAGE_Plugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* TRANS_IMAGE_Plugin::clone() const
{
    return new TRANS_IMAGE_Plugin(mLayerName,mImageHeight,mImageWidth,mLayout,mColorOrder);
}

void TRANS_IMAGE_Plugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* TRANS_IMAGE_Plugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

TRANS_IMAGE_Creator::TRANS_IMAGE_Creator()
{
    mPluginAttributes.emplace_back(PluginField("image_heigth", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("image_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("layout", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("color_order", nullptr, PluginFieldType::kINT32, 1));
    
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* TRANS_IMAGE_Creator::getPluginName() const
{
    return TRANS_IMAGE_PLUGIN_NAME;
}

const char* TRANS_IMAGE_Creator::getPluginVersion() const
{
    return TRANS_IMAGE_PLUGIN_VERSION;
}

const PluginFieldCollection* TRANS_IMAGE_Creator::getFieldNames()
{
    return &mFC;
}

IPluginV2* TRANS_IMAGE_Creator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    PluginFormat layout=PluginFormat::kNCHW;
    ColorOrder color_order=ColorOrder::kRGB;
    int image_height=600;
    int image_width=600;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 5);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "image_height") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            image_height = *(static_cast<const float*>(fields[i].data));
        }else if (strcmp(fields[i].name, "image_width") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            image_width = *(static_cast<const float*>(fields[i].data));
        }else if (strcmp(fields[i].name, "layout") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            layout=static_cast<PluginFormat>(*(static_cast<const float*>(fields[i].data)));
        }else if (strcmp(fields[i].name, "color_order") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            color_order =static_cast<ColorOrder>(*(static_cast<const float*>(fields[i].data)));
        }
    }
    TRANS_IMAGE_Plugin* obj = new TRANS_IMAGE_Plugin(name,image_height,image_width,layout,color_order);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* TRANS_IMAGE_Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    TRANS_IMAGE_Plugin* obj = new TRANS_IMAGE_Plugin(name, serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void TRANS_IMAGE_Creator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* TRANS_IMAGE_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
