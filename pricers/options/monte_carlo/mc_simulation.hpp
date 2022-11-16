// 2022/11/16
// author: Hyung Jip Lee
// Monte Carlo Simulation for Option Pricing

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>

normal_distribution<double> N(0.0, 1.0); // standard
student_t_distribution<double> t_dist(5); // input is degree of freedom

class GBM {
    
}