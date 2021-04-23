#include "NvInfer.h"
#include "tensorrt/trt_utils.h"

using namespace nvinfer1;
using namespace quake::framework::ops_lib;

class test_1{
public:
  //basic APIs for testing
  bool build(TRTUniquePtr<IBuilder>& builder,TRTUniquePtr<INetworkDefinition>& network,TRTUniquePtr<IBuilderConfig>& config,
    ITensor** inputs,ITensor** outputs,int batch_size,DLRLogger& dlr_logger);
  bool clean_up();

private:
  std::map<std::string, Weights> mWeightsMap;
};

