// 2022/11/16
// author: Hyung Jip Lee
// Monte Carlo Simulation for Option Pricing
#ifndef MC_SIMULATION_HPP
#define MC_SIMULATION_HPP

#include <math.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <string>
#include <algorithm>
// #include <boost/math/distributions/inverse_gaussian.hpp>

std::default_random_engine gen;
std::normal_distribution<double> N(0.0, 1.0);
// supposed to use norminv from boost, fix this

// option pricer using standard monte carlo
double Option_StdMC(std::string callorput, double S, double X,
    double T, double r, double b, double sig, long n) {

        double sum;
        double sample;
        double St;
        int z;

        double price; // final result

        double drift = (b - pow(sig, 2)/2) * T;

        if (callorput == "c") {
            z = 1;
        } else if (callorput == "p") {
            z = -1;
        }

        for (int i = 0; i < n; i++) {
            sample = N(gen);
            St = S * exp(drift + sig*sqrt(T) * sample);
            sum += std::max(z * (St - X), 0.0);
        }

        price = exp(-r * T) * sum / n;

}

#endif