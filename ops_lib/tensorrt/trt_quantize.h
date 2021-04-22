//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_TRT_QUANTIZE_H
#define OPSLIB_TRT_QUANTIZE_H

#include "trt_utils.h"
#include <ctime>

using namespace nvinfer1;
using namespace std;

namespace quake {
namespace framework {
namespace ops_lib {

class DLRInt8CalibratorUtils
{
public:
  DLRInt8CalibratorUtils(DLRBatchStream& stream,int firstBatch,const std::string& network_name,bool readCache=true);

  ~DLRInt8CalibratorUtils() {CHECK(cudaFree(mDeviceInput));}

  int get_batchsize() {return mStream.getBatchSize();}

  bool get_batch(void* bindings[], const char* names[], int nbBindings);

  const void* read_calibration_cache(size_t& length);

  void write_calibration_cache(const void* cache, size_t length);

  std::string calibration_table_name(){ return mNetworkName+".range"; }

private:
  DLRBatchStream mStream;
  std::string mNetworkName;
  bool mReadCache{true};
  size_t mInputCount;
  void* mDeviceInput{nullptr};
  std::vector<char> mCalibrationCache;
};

#define DEFAULT_CALIBRATOR_FUNCS(cal_cls) \
public:                                                                                                   \
  cal_cls(DLRBatchStream& stream,int firstBatch,const std::string& network_name,bool readCache=true) {  \
    mCalibratorUtils.reset(new DLRInt8CalibratorUtils(stream,firstBatch,network_name,readCache));       \
  };                                                                                                      \
                                                                                                          \
  virtual ~cal_cls() {}                                                                                   \
                                                                                                          \
  int getBatchSize() const override {return mCalibratorUtils->get_batchsize();}                           \
                                                                                                          \
  bool getBatch(void* bindings[], const char* names[], int nbBindings) override {                         \
    return mCalibratorUtils->get_batch(bindings,names,nbBindings);                                        \
  };                                                                                                      \
                                                                                                          \
  const void* readCalibrationCache(size_t& length) override {                                             \
    return mCalibratorUtils->read_calibration_cache(length);                                              \
  };                                                                                                      \
                                                                                                          \
  void writeCalibrationCache(const void* cache, size_t length) override {                                 \
    return mCalibratorUtils->write_calibration_cache(cache,length);                                       \
  };                                                                                                      \
                                                                                                          \
private:                                                                                                  \
  std::unique_ptr<DLRInt8CalibratorUtils> mCalibratorUtils;                                             \
  std::string calibrationTableName(){                                                                     \
    return mCalibratorUtils->calibration_table_name(); }                                                  \

//define the calibrators
class DLRInt8EntropyCalibrator : public IInt8EntropyCalibrator
{
DEFAULT_CALIBRATOR_FUNCS(DLRInt8EntropyCalibrator)
};

class DLRInt8EntropyCalibrator2 : public IInt8EntropyCalibrator2
{
DEFAULT_CALIBRATOR_FUNCS(DLRInt8EntropyCalibrator2)
};

class DLRInt8MinMaxCalibrator : public IInt8MinMaxCalibrator
{
DEFAULT_CALIBRATOR_FUNCS(DLRInt8MinMaxCalibrator)
};

class DLRInt8LegacyCalibrator : public IInt8LegacyCalibrator
{
DEFAULT_CALIBRATOR_FUNCS(DLRInt8LegacyCalibrator)
public:
  DLRInt8LegacyCalibrator(DLRBatchStream& stream,int firstBatch,const std::string& network_name,
    double cutoff,double quantile,bool readCache=true);

  double getQuantile() const override {return mQuantile;};

  double getRegressionCutoff() const override {return mCutoff;};

  const void* readHistogramCache(size_t& length) override;

  void writeHistogramCache(const void* cache, size_t length) override;

private:
  std::vector<char> mHistogramCache;
  double mCutoff{1.0};
  double mQuantile{0.99999};
};

void setLayerPrecision(TRTUniquePtr<INetworkDefinition>& network,DataType dtype,DLRLogger& logger);

void setAllTensorScales(TRTUniquePtr<INetworkDefinition>& network,float inScales,float outScales);

bool setDynamicRange(const std::string& file,TRTUniquePtr<INetworkDefinition>& network,DLRLogger& logger);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_TRT_QUANTIZE_H
