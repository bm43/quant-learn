// optimization of sharpe ratio for delta hedging strategy
// https://deliverypdf.ssrn.com/delivery.php?ID=629021065069064010018111076124016072004042024048051009122064090096090118110024125092123124006123042032124102105123123017105070119033078019018121022010019109092096004070050017097122097075113004082006087111113024092013112086098117094123077104071118001024&EXT=pdf&INDEX=TRUE


#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Function to calculate the delta of an option
double optionDelta(double S, double K, double r, double sigma, double T, bool isCall) {
    double d1 = (log(S/K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    if(isCall)
      return normcdf(d1);
    else
      return normcdf(d1) - 1;
}

// Function to calculate the PnL of the delta hedging strategy
vector<double> PnL(double S, double K, double r, double sigma, double T, bool isCall, double delta, double tc, double hf) {
    double dt = T/hf;
    double S_ = S;
    double delta_ = delta;
    double PnL_ = 0;
    double sharpeRatio_ = 0;
    vector<double> result;
    for (int i = 0; i < hf; i++) {
        double dS = S_ * sigma * sqrt(dt) *  norminv(rand());
        S_ += dS;
        double C = optionDelta(S_, K, r, sigma, T, isCall) * S_ - delta_ * S_;
        PnL_ += dS - C - tc;
        delta_ = optionDelta(S_, K, r, sigma, T, isCall);
    }
    sharpeRatio_ = PnL_ / sqrt(T);
    result.push_back(PnL_);
    result.push_back(sharpeRatio_);
    return result;
}

int main() {
    double S = 100; // Underlying asset price
    double K = 110; // Strike price
    double r = 0.05; // Risk-free rate
    double sigma = 0.2; // Volatility
    double T = 1; // Time to expiration (in years)
    bool isCall = true; // Whether the option is a call or a put option
    double tc = 0.01; // transaction cost
    double delta = optionDelta(S, K, r, sigma, T, isCall);
    double hf = 1; // initial hedging frequency
    double max_sharpe = 0;
    double max_hf = 0;
    for (int i = 1; i <= 100; i++) {
        hf = i;
        vector<double> result = PnL(S, K, r, sigma, T, isCall, delta, tc, hf);
        if(result[1] > max_sharpe){
            max_sharpe = result[1];
            max_hf = hf;
        }
    }
    cout << "The optimal hedging frequency is: " << max_hf << endl;
    cout << "The maximum Sharpe ratio is: " << max_sharpe << endl;
    return 0;
}