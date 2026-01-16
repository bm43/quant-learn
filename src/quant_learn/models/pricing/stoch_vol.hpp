#include <iostream>
#include <cmath>
#include <random>

class StochasticVolatilityModel {
    public:
        StochasticVolatilityModel(double S0, double mu, double kappa, double theta, double xi, double rho)
            : S0(S0), mu(mu), kappa(kappa), theta(theta), xi(xi), rho(rho) {}

        double simulatePrice(int steps) {
            double S = S0;
            double v = theta;
            std::default_random_engine generator;
            std::normal_distribution<double> normal_dist(0.0, 1.0);

            for (int i = 0; i < steps; i++) {
                double z1 = normal_dist(generator);
                double z2 = rho * z1 + sqrt(1 - rho * rho) * normal_dist(generator);
                v = v + kappa * (theta - v) * dt + xi * sqrt(v * dt) * z1;
                S = S * exp(mu * dt + sqrt(v * dt) * z2);
            }
            return S;
        }

    private:
        double S0;
        double mu;
        double kappa;
        double theta;
        double xi;
        double rho;
        double dt = 1;
};