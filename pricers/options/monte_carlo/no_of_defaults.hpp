#ifndef NO_OF_DEFAULTS_HPP
#define NO_OF_DEFAULTS_HPP

// models the number of defaults in a heterogeneous portfolio

#include <iostream>
#include <random>

int simulate_defaults(std::vector<double> pd, std::vector<double> LGD) {
    int num_defaults = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < pd.size(); i++) {
        std::bernoulli_distribution dist(pd[i]);
        if (dist(gen) == 1) {
            num_defaults++;
            // assuming LGD is given as percentage
            LGD[i] = LGD[i]/100;
        }
    }
    return num_defaults;
}

double calculate_expected_loss(std::vector<double> pd, std::vector<double> LGD, std::vector<double> exposures) {
    double expected_loss = 0;
    int num_defaults = simulate_defaults(pd, LGD);
    for (int i = 0; i < num_defaults; i++) {
        expected_loss += LGD[i] * exposures[i];
    }
    return expected_loss;
}

int main() {
    std::vector<double> pd = { 0.01, 0.02, 0.03 };
    std::vector<double> LGD = { 40, 50, 60 };
    std::vector<double> exposures = { 100000, 200000, 300000 };
    int num_simulations = 10000;

    double total_expected_loss = 0;
    for (int i = 0; i < num_simulations; i++) {
        total_expected_loss += calculate_expected_loss(pd, LGD, exposures);
    }
    double expected_loss = total_expected_loss / num_simulations;
    std::cout << "Expected loss: " << expected_loss << std::endl;

    return 0;
}