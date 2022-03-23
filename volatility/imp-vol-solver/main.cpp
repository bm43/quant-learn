#include "black_scholes.h"
#include "interval_bisection.h"
#include "brents_method.h"
#include <iostream>

int main(int argc, char **argv) {
  // First we create the parameter list
  double S = 100.0;  // Underlying spot price
  double K = 100.0;  // Strike price
  double r = 0.05;   // Risk-free rate (5%)
  double T = 1.0;    // One year until expiry
  double C_M = 10.5; // Option market price
  // Q: why define the option price when you have the
  // call_price function that calculates it??
  
  BlackScholesCall bsc(S,K,r,T);
  

  // interval bisection params
  double low_vol = 0.05;
  double high_vol = 0.35;
  double epsilon = 0.001;

  double sigma = interval_bisection(C_M, low_vol, high_vol, epsilon, bsc);
  double sigma2 = brents_method(C_M, low_vol, high_vol, epsilon, bsc);
  std::cout<<"\n"<<"Implied vol: "<<sigma<<"\n";
  std::cout<<"Implied vol2: "<<sigma2;
  return 0;
}
