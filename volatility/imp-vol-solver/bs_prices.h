#ifndef __BS_PRICES_H
#define __BS_PRICES_H
// if the token in #ifndef <token>
// is not defined, define it as follows
// purpose is just to define a header once,
// no need to define it over and over

#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>

// normal proba density func
// input double, output double
/*
double norm_pdf(const double x) {
  return (1.0/(pow(2*M_PI,0.5)))*exp(-0.5*x*x);
}
*/

double norm_cdf(const double x) {
  return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// components of the black scholes solution:
// d_2 = d-
// d_1 = d+
double d_j(const int j, const double S, const double K, const double r, const double sigma, const double T) {
  return (log(S/K) + (r + (pow(-1,j-1))*0.5*sigma*sigma)*T)/(sigma*(pow(T,0.5)));
}

// option price compute using black scholes solution:
// https://www.investopedia.com/articles/optioninvestor/07/options_beat_market.asp
double call_price(const double S, const double K, const double r, const double sigma, const double T) {
  return S * norm_cdf(d_j(1, S, K, r, sigma, T))-K*exp(-r*T) * norm_cdf(d_j(2, S, K, r, sigma, T));
}

double put_price(const double S, const double K, const double r, const double sigma, const double T) {
  // p(0) = e−rT KN(−d2) − S(0)N(−d1) at time 0
  return exp(-r*T) * K * norm_cdf(-d_j(2, S, K, r, sigma, T)) - S * norm_cdf(-d_j(1, S, K, r, sigma, T));
}

double call_vega(const double S, const double K, const double r, const double sigma, const double T) {
  return (1/sqrt(2*M_PI)) * S * exp(-pow(d_j(1, S, K, r, sigma, T),2)/2);
}

#endif
