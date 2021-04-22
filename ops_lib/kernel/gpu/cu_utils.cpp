//
// Created by chengjin on 2020-06-02.
//

#include "cu_utils.h"

namespace quake {
namespace framework {
namespace ops_lib {

int n_blocks(int size, int block_size) {
  return size / block_size + ((size % block_size == 0)? 0 : 1);
}

int get_block_grid_1d(const int size){
  int dimGrid = n_blocks(size,CU1DBLOCK);
  if (dimGrid > 256) {
    dimGrid = 256;
  }
  return dimGrid;
}

MatrixDim get_matrix_dim(int rows,int cols,int stride){
  MatrixDim d;
  d.rows=rows;
  d.cols=cols;
  d.stride=(stride==-1)?cols:stride;
  return d;
}

__half dlr_float2half(float f){
  uint32_t x = bitwise_cast<uint32_t, float>(f);
  uint32_t u = (x & 0x7fffffff);

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000)
      return bitwise_cast<__half, uint16_t>(uint16_t(0x7fff));

  uint16_t sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff)
      return bitwise_cast<__half, uint16_t>(sign | uint16_t(0x7c00));

  if (u < 0x33000001)
      return bitwise_cast<__half, uint16_t>(sign | uint16_t(0x0000));

  uint32_t exponent = ((u >> 23) & 0xff);
  uint32_t mantissa = (u & 0x7fffff);

  uint32_t shift;
  if (exponent > 0x70)
  {
      shift = 13;
      exponent -= 0x70;
  }
  else
  {
      shift = 0x7e - exponent;
      exponent = 0;
      mantissa |= 0x800000;
  }

  uint32_t lsb = (1 << shift);
  uint32_t lsb_s1 = (lsb >> 1);
  uint32_t lsb_m1 = (lsb - 1);

  // Round to nearest even.
  uint32_t remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if ((remainder > lsb_s1) || ((remainder == lsb_s1) && (mantissa & 0x1)))
  {
      ++mantissa;
      if (!(mantissa & 0x3ff))
      {
          ++exponent;
          mantissa = 0;
      }
  }

  return bitwise_cast<__half, uint16_t>(sign | uint16_t(exponent << 10) | uint16_t(mantissa));
}

float dlr_half2float(__half h){
  uint16_t x = bitwise_cast<uint16_t, __half>(h);
  uint32_t sign = ((x >> 15) & 1);
  uint32_t exponent = ((x >> 10) & 0x1f);
  uint32_t mantissa = (static_cast<uint32_t>(x & 0x3ff) << 13);

  if (exponent == 0x1f)
  { /* NaN or Inf */
      if (mantissa != 0)
      { // NaN
          sign = 0;
          mantissa = 0x7fffff;
      }
      else // Inf
          mantissa = 0;
      exponent = 0xff;
  }
  else if (!exponent)
  { /* Denorm or Zero */
      if (mantissa)
      {
          unsigned int msb;
          exponent = 0x71;
          do
          {
              msb = (mantissa & 0x400000);
              mantissa <<= 1; /* normalize */
              --exponent;
          } while (!msb);
          mantissa &= 0x7fffff; /* 1.mantissa is implicit */
      }
  }
  else
      exponent += 0x70;
  return bitwise_cast<float, uint32_t>((sign << 31) | (exponent << 23) | mantissa);
}

} // namespace ops_lib
} // namespace framework
} // namespace quake
