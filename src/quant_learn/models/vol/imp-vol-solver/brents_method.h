#ifndef __BRENTS_METHOD_H
#define __BRENTS_METHOD_H
#include <cmath>
#include <iostream>
template <typename T>
double brents_method (double y_target, double a, double b, double TOL, T f) {
    double fa = f(a);
    double fb = f(b);
    std::cout<<"fa: "<<fa<<"\n"<<"fb: "<<fb;
    if ( !( (fa < y_target < fb) || (fa > y_target > fb) ) ) {
        throw std::invalid_argument("The root is not bracketed");
        return 0;
    }
    if (fabs(fa) < fabs(fb)) {
        std::swap(a,b);
    }

    double c = a;
    double fc = f(c);
    bool mflag = true; // mlflag = True means flag is set, False means flag is cleared
    double s = 0.0;
    double d = 0.0;
    do {
        if ( (fa != f(c)) && (fb != f(c)) ) {
            s = ( a * fb * fc / ((fa - fb) * (fa - fc)) )
                + ( b * fa * fc / ((fb - fa) * (fb - fc)) )
                + ( c * fa * fb / ((fc - fa) * (fc - fb)) );
        } else {
            s = b - fb * (b - a) / (fb - fa); // secant method (?)
        }

        if (( (s < (3 * a + b) * 0.25) || (s > b) ) ||
            ( mflag && (std::abs(s-b) >= (std::abs(b-c) * 0.5)) ) ||
            ( !mflag && (std::abs(s-b) >= (std::abs(c-d) * 0.5)) ) ||
            ( mflag && (std::abs(b-c) < TOL) ) ||
            ( !mflag && (std::abs(c-d) < TOL))  ) {
                s = 0.5*(a+b);
                mflag = true;
            } else {
                mflag = false;
            }

        double fs = f(s);
        double d = c;
        c = b;

        if (fa*fs < 0){
            b = s;
        } else {
            a = s;
        }

        if ( fabs(fa) < fabs(fb) ) {
            std::swap(a,b);
        }
    } while ( ( (fb==0) || (f(s)==0) ) || (fabs(b-a) < TOL) );

    if (f(s) == 0) {
        return s;
    } else {
        return b;
    }

}

#endif
