#ifndef ANNUAL_TREASURY_HPP
#define ANNUAL_TREASURY_HPP

double annual_treasury(double FV, double couponRate, double yield, int ytm) {
    double couponPayment = FV * couponRate; // calculate annual coupon payment
    double bondPrice = 0;

    for (int i = 1; i <= ytm; i++) {
        bondPrice += couponPayment / pow(1 + yield, i); // add present value of each coupon payment
    }

    bondPrice += FV / pow(1 + yield, ytm); // add present value of face value

    return bondPrice;
}

#endif