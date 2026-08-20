#ifndef INTERVAL_BISECTION_H
#define INTERVAL_BISECTION_H

#include <cmath>
#include <stdexcept>

// Solve g(x) = y_target on [m, n] by halving the bracket.
template <typename T>
double interval_bisection(double y_target, double m, double n, double epsilon, T g) {
  if ((g(m) - y_target) * (g(n) - y_target) >= 0.0) {
    throw std::invalid_argument("root is not bracketed");
  }
  double x = 0.5 * (m + n);
  for (int i = 0; i < 200; ++i) {
    double fx = g(x) - y_target;
    if (std::fabs(fx) < epsilon) break;
    if ((g(m) - y_target) * fx < 0.0) n = x; else m = x;
    x = 0.5 * (m + n);
  }
  return x;
}

#endif