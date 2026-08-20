#ifndef NEWTON_RAPHSON_HPP
#define NEWTON_RAPHSON_HPP

#include <cmath>

// Solve g(x) = y_target using the derivative g_prime.
template <typename T, typename E>
double newton_raphson(double y_target, double guess, double epsilon, T g, E g_prime) {
  double x = guess;
  for (int i = 0; i < 100; ++i) {
    double step = (g(x) - y_target) / g_prime(x);
    x -= step;
    if (std::fabs(step) < epsilon) break;
  }
  return x;
}

#endif