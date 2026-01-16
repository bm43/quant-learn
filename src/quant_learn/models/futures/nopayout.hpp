#include <cmath>
using namespace std;

double futures_price(double& S, double& r, double& time_to_maturity) {
    // S: current price of asset
    // r: interest rate
    return exp(r * time_to_maturity) * S;
}