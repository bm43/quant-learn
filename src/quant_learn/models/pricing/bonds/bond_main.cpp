#include <iostream>
#include <cmath>
#include "bondpricing.hpp"

using namespace std;

int main() {
    double faceValue, couponRate, yield;
    int yearsToMaturity;

    cout << "Enter face value: ";
    cin >> faceValue;

    cout << "Enter coupon rate (as decimal): ";
    cin >> couponRate;

    cout << "Enter yield (as decimal): ";
    cin >> yield;

    cout << "Enter years to maturity: ";
    cin >> yearsToMaturity;

    double bondPrice = bondPrice(faceValue, couponRate, yield, yearsToMaturity);

    cout << "Bond price: " << bondPrice << endl;

    return 0;
}