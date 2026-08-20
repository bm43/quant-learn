#include "black_scholes.h"
#include "bs_prices.h"
#include "interval_bisection.h"
#include "brents_method.h"
#include "newton_raphson.hpp"
#include <iostream>

int main() {
  const double S = 100.0, K = 100.0, r = 0.05, T = 1.0;
  const double market_price = 10.5;

  BlackScholesCall call(S, K, r, T);
  CallVega vega(S, K, r, T);

  std::cout << "market price     " << market_price << "\n"
            << "bisection        " << interval_bisection(market_price, 0.01, 1.0, 1e-8, call) << "\n"
            << "brent            " << brents_method(market_price, 0.01, 1.0, 1e-8, call) << "\n"
            << "newton-raphson   " << newton_raphson(market_price, 0.50, 1e-8, call, vega) << "\n";
  return 0;
}