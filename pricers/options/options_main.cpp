#include "merton_jump_diff.hpp"
#include <iostream>

using namespace std;

int main() {
    double S = 100.0;  // Underlying spot price
    double K = 100.0;  // Strike price
    double r = 0.05;   // Risk-free rate (5%)
    double T = 1.0;    // One year until expiry
    double sigma = 10.5; // volatility of asset
    double delta = 1;
    double kappa = 123;
    double lambda = 34;
    cout<<merton_jump_diffusion_call(S, K, T, r, sigma, delta, kappa, lambda);
    return 0;
}