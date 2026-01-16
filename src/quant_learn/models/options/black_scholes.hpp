#include <cmath>

#define M_SQRT1_2 0.707106781186547524401

double norm_cdf(const double x) {
  return 0.5 * std::erfc(-x * M_SQRT1_2);
}

double d_j(const int j, const double S, const double K, const double r, const double sigma, const double T) {
  return (log(S/K) + (r + (pow(-1,j-1))*0.5*sigma*sigma)*T)/(sigma*(pow(T,0.5)));
}

double black_scholes_call_price(const double S, const double K, const double r, const double sigma, const double T) {
  return S * norm_cdf(d_j(1, S, K, r, sigma, T))-K*exp(-r*T) * norm_cdf(d_j(2, S, K, r, sigma, T));
}