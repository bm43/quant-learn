#ifndef __BLACK_SCHOLES_H
#define __BLACK_SCHOLES_H

class BlackScholesCall {

    private:
        double S; // underlying asset price cuz its option
        double K; //strike price
        double r; // risk-free rate
        double T; // time to maturity

    public:
        BlackScholesCall(double _S, double _K,
        double _r, double _T);
        double operator()(double sigma) const;
};

class BlackScholesPut {

    private:
        double S; // underlying asset price cuz its option
        double K; //strike price
        double r; // risk-free rate
        double T; // time to maturity

    public:
        BlackScholesPut(double _S, double _K,
        double _r, double _T);
        double operator()(double sigma) const;
};

// vega:
class CallVega {

    private:
        double S; // underlying asset price cuz its option
        double K; //strike price
        double r; // risk-free rate
        double T; // time to maturity
    public:
        CallVega(double _S, double _K,
        double _r, double _T);
        double operator()(double sigma) const;
};

#endif
