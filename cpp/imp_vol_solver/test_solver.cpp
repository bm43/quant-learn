#include "black_scholes.h"
#include "bs_prices.h"
#include "interval_bisection.h"
#include "brents_method.h"
#include "newton_raphson.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

static int failures = 0;

void check(const char* name, double got, double want, double tol) {
  bool ok = std::fabs(got - want) < tol;
  std::cout << (ok ? "PASS  " : "FAIL  ") << name
            << "  got " << got << "  want " << want << "\n";
  if (!ok) ++failures;
}

int main() {
  const double S = 100.0, K = 95.0, r = 0.05, T = 2.0;
  const double true_sigma = 0.23;

  // Price an option at a known vol, then recover that vol from the price.
  const double price = call_price(S, K, r, true_sigma, T);
  BlackScholesCall call(S, K, r, T);
  CallVega vega(S, K, r, T);

  check("bisection", interval_bisection(price, 0.01, 1.0, 1e-8, call), true_sigma, 1e-4);
  check("brent",     brents_method(price, 0.01, 1.0, 1e-8, call),      true_sigma, 1e-4);
  check("newton",    newton_raphson(price, 0.50, 1e-8, call, vega),    true_sigma, 1e-4);

  // Vega should match a finite difference of the price.
  const double h = 1e-5;
  const double fd = (call_price(S, K, r, true_sigma + h, T)
                   - call_price(S, K, r, true_sigma - h, T)) / (2 * h);
  check("vega", call_vega(S, K, r, true_sigma, T), fd, 1e-4);

  // Put-call parity: C - P = S - K exp(-rT)
  const double parity = call_price(S, K, r, true_sigma, T)
                      - put_price(S, K, r, true_sigma, T);
  check("put-call parity", parity, S - K * std::exp(-r * T), 1e-10);

  std::cout << (failures ? "\nFAILED\n" : "\nAll tests passed\n");
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}