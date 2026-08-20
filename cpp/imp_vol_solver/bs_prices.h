#ifndef BS_PRICES_H
#define BS_PRICES_H

#define _USE_MATH_DEFINES
#include <cmath>

inline double norm_cdf(double x) {
  return 0.5 * std::erfc(-x * M_SQRT1_2);
}

inline double norm_pdf(double x) {
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// j = 1 gives d+, j = 2 gives d-
inline double d_j(int j, double S, double K, double r, double sigma, double T) {
  double sign = (j == 1) ? 1.0 : -1.0;
  return (std::log(S / K) + (r + sign * 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}

inline double call_price(double S, double K, double r, double sigma, double T) {
  return S * norm_cdf(d_j(1, S, K, r, sigma, T))
       - K * std::exp(-r * T) * norm_cdf(d_j(2, S, K, r, sigma, T));
}

inline double put_price(double S, double K, double r, double sigma, double T) {
  return K * std::exp(-r * T) * norm_cdf(-d_j(2, S, K, r, sigma, T))
       - S * norm_cdf(-d_j(1, S, K, r, sigma, T));
}

// vega = S * sqrt(T) * phi(d+)
inline double call_vega(double S, double K, double r, double sigma, double T) {
  return S * std::sqrt(T) * norm_pdf(d_j(1, S, K, r, sigma, T));
}

#endif