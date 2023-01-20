#include "no_of_defaults.hpp"
#include "mc_simulation.hpp"

int main() {

    // main func for number of defaults model
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