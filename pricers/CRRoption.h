<<<<<<< HEAD
#ifndef CRROPTION_H
#define CRROPTION_H

//inputting and displaying option data
int GetInputData(int& N, double& K);

//pricing European option
double PriceByCRR(double S0, double U, double D, double R, int N, double K);

//computing call payoff
double CallPayoff(double z, double K);

=======
#ifndef CRROPTION_H
#define CRROPTION_H

//inputting and displaying option data
int GetInputData(int& N, double& K);

//pricing European option
double PriceByCRR(double S0, double U, double D, double R, int N, double K);

//computing call payoff
double CallPayoff(double z, double K);

>>>>>>> d2b34c2a0baed62724d4653b460d67ca50e1a50e
#endif