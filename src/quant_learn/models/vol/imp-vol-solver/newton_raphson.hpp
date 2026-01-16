#ifndef NEWTON_RAPHSON_HPP
#define NEWTON_RAPHSON_HPP

template<typename T, typename E>

double newton_raphson(double y_target, double guess, double epsilon, T g, E g2) {
    // init values
    double x_prev = guess;
    double x_next = x_prev - (g(x_prev) - y_target)/g2(x_prev);
    while (x_next - x_prev > epsilon || x_prev - x_next > epsilon) {
        x_prev = x_next;
        x_next = x_prev - (g(x_prev) - y_target)/g2(x_prev);
    }
    return x_next;
}

#endif