#ifndef LOG_NORMAL_DIFFUSION_HPP
#define LOG_NORMAL_DIFFUSION_HPP

#include <iostream>
#include <cmath>
#include <random>
#include <vector>

class LogNormalModel {
    public:
        LogNormalModel(std::vector<double> S0, std::vector<double> mu, std::vector<std::vector<double>> sigma, double dt)
            : S0(S0), mu(mu), sigma(sigma), dt(dt) {}
        
        std::vector<double> simulatePrice(int steps) {
            std::vector<double> S = S0;
            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0, 1.0);

            for (int i = 0; i < steps; i++) {
                std::vector<double> z(S.size());
                for(int j=0;j<S.size();j++)
                {
                    z[j] = distribution(generator);
                }
                for(int j=0;j<S.size();j++)
                {
                    S[j] = S[j] * exp((mu[j] - 0.5 * sigma[j][j] * sigma[j][j]) * dt + sigma[j][j] * sqrt(dt) * z[j]);
                }
            }
            return S;
        }

    private:
        std::vector<double> S0;
        std::vector<double> mu;
        std::vector<std::vector<double>> sigma;
        double dt;
};

#endif