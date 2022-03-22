#ifndef __BRENTS_METHOD_H
#define __BRENTS_METHOD_H
#include <cmath>
#include <iostream>
template <typename T>
double brents_method (double a, double b, double tol, T f) {
    double fa = f(a);
    double fb = f(b);
    
    if (!(fa*fb < 0)) {
        throw std::invalid_argument("The root is not bracketed");
        return 0;
    }
    if (fabs(fa) < fabs(fb)) {
        double dummy = a;
        a = b;
        b = dummy;
    }
    double c = a;
    bool mflag = true; // mlflag = True means flag is set, False means flag is cleared
    do {
        if (fa != f(c)) && (fb != f(c)) {
            s = ( a * fb * fc / ((fa - fb) * (fa - fc)) )
                + ( b * fa * fc / ((fb - fa) * (fb - fc)) )
                + ( c * fa * fb / ((fc - fa) * (fc - fb)) );
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }
    } while ((fb)==0) || (f(s)==0)) || (fabs(b-a) < tol)
}


#endif