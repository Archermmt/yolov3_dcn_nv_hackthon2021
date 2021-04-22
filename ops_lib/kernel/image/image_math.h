//
// Created by chengjin on 2020-06-02.
//

#ifndef OPSLIB_KERNEL_IMAGEMATH_H
#define OPSLIB_KERNEL_IMAGEMATH_H

#include <opencv2/opencv.hpp>

namespace quake {
namespace framework {
namespace ops_lib {

cv::Mat similarTransform(cv::Mat src,cv::Mat dst);

} // namespace ops_lib
} // namespace framework
} // namespace quake

#endif //OPSLIB_KERNEL_IMAGEMATH_H
