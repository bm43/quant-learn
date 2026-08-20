#ifndef BRENTS_METHOD_H
#define BRENTS_METHOD_H

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Solve g(x) = y_target on [a, b]. Inverse quadratic interpolation where it
// helps, bisection where it does not.
template <typename T>
double brents_method(double y_target, double a, double b, double tol, T g) {
  double fa = g(a) - y_target;
  double fb = g(b) - y_target;
  if (fa * fb >= 0.0) throw std::invalid_argument("root is not bracketed");
  if (std::fabs(fa) < std::fabs(fb)) { std::swap(a, b); std::swap(fa, fb); }

  double c = a, fc = fa, d = b - a;
  bool used_bisection = true;

  for (int i = 0; i < 100 && fb != 0.0 && std::fabs(b - a) > tol; ++i) {
    double s;
    if (fa != fc && fb != fc) {
      s = a * fb * fc / ((fa - fb) * (fa - fc))
        + b * fa * fc / ((fb - fa) * (fb - fc))
        + c * fa * fb / ((fc - fa) * (fc - fb));
    } else {
      s = b - fb * (b - a) / (fb - fa);
    }

    double lo = std::min((3.0 * a + b) / 4.0, b);
    double hi = std::max((3.0 * a + b) / 4.0, b);
    bool bisect =
        (s < lo || s > hi)
        || (used_bisection  && std::fabs(s - b) >= std::fabs(b - c) / 2.0)
        || (!used_bisection && std::fabs(s - b) >= std::fabs(c - d) / 2.0)
        || (used_bisection  && std::fabs(b - c) < tol)
        || (!used_bisection && std::fabs(c - d) < tol);

    if (bisect) { s = 0.5 * (a + b); used_bisection = true; }
    else        { used_bisection = false; }

    double fs = g(s) - y_target;
    d = c; c = b; fc = fb;
    if (fa * fs < 0.0) { b = s; fb = fs; } else { a = s; fa = fs; }
    if (std::fabs(fa) < std::fabs(fb)) { std::swap(a, b); std::swap(fa, fb); }
  }
  return b;
}

#endif