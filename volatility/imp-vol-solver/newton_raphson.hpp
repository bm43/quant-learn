#ifndef NEWTON_RAPHSON_HPP
#define NEWTON_RAPHSON_HPP

template<typename T>
template<typename T2>
double newton_raphson(T g, T2 g2, double Tgt, double guess, double epsilon) {
    // init values
    double x_prev = guess;
    double x_next = x_prev - (g(x_prev) - Tgt)/g2(x_prev);
    while (x_next - x_prev > epsilon || x_prev - x_next > epsilon) {
        x_prev = x_next;
        x_next = x_prev - (g(x_prev) - Tgt)/g2(x_prev);
    }
    return x_next;
}

#endif