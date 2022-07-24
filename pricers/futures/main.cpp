#include "nopayout.hpp"
#include <cmath>

int main() {
    double S = 100;
    double r = 0.1;
    double t_to_mat = 5;
    futures_price(S, r, t_to_mat);
    return 0;
}