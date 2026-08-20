#ifndef __BLACK_SCHOLES_CPP
#define __BLACK_SCHOLES_CPP

#include "black_scholes.h"
#include "bs_prices.h"

BlackScholesCall::BlackScholesCall(double _S, double _K, double _r, double _T): S(_S), K(_K), r(_r), T(_T) {

}

// https://docs.microsoft.com/en-us/cpp/standard-library/function-objects-in-the-stl?view=msvc-170
double BlackScholesCall::operator()(double sigma) const {
    return call_price(S,K,r,sigma,T);
}

BlackScholesPut::BlackScholesPut(double _S, double _K, double _r, double _T): S(_S), K(_K), r(_r), T(_T) {

}

double BlackScholesPut::operator()(double sigma) const {
    return put_price(S,K,r,sigma,T);
}

// vega:
CallVega::CallVega(double _S, double _K, double _r, double _T): S(_S), K(_K), r(_r), T(_T) {
    
}

double CallVega::operator()(double sigma) const {
    return call_vega(S,K,r,sigma,T);
}

#endif
