#ifndef JUMP_DIFFUSION_HPP
#define JUMP_DIFFUSION_HPP

#include <iostream>
#include <cmath>
#include <random>

class JumpDiffusionModel {
    public:
        JumpDiffusionModel(double S0, double mu, double sigma, double lambda, double k, double theta)
            : S0(S0), mu(mu), sigma(sigma), lambda(lambda), k(k), theta(theta) {}
        
        double simulatePrice(int steps) {
            double S = S0;
            std::default_random_engine generator;
            std::normal_distribution<double> normal_dist(0.0, 1.0);
            std::exponential_distribution<double> exp_dist(1.0);
            for (int i = 0; i < steps; i++) {
                double z = normal_dist(generator);
                double jump = k * (exp_dist(generator) - theta);
                S = S * exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * z + jump);
                if (exp_dist(generator) < lambda * dt) {
                    S = S * exp(jump);
                }
            }
            return S;
        }

    private:
        double S0;
        double mu;
        double sigma;
        double lambda;
        double k;
        double theta;
        double dt = 1;
};

#endif