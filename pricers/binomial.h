#ifndef BINOMIAL_H
#define BINOMIAL_H

double RiskNeutProb(double U, double D, double R);

double S(double S0, double U, double D, int n, int i);

int GetInputData(double& S0, double& U, double& D, double& R);

#endif