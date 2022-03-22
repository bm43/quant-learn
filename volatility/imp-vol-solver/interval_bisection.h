#ifndef __INTERVAL_BISECTION_H
#define __INTERVAL_BISECTION_H
// a root finding algorithm, this will be applied to 
#include <cmath>
// root that we are trying to find here is the implied volatility
#include <iostream>
// function template T
template<typename T>
// function object type T
double interval_bisection(double y_target, double m, double n, double epsilon, T g) {
    double x = 0.5*(m+n);
    double y = g(x);

    do {

        if (y < y_target){
            m=x;
        }

        if (y > y_target){
            n=x;
        }

        x = 0.5*(m+n);
        y = g(x);
        // std::cout<<"diff: "<<fabs(y-y_target)<<"\n";
        // std::cout<<"epsilon: "<<epsilon<<"\n";
    } while (fabs(y-y_target) > epsilon);
    return x;
}
#endif