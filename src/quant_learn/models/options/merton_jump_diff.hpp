#include <cmath>
#include "black_scholes.hpp"


double merton_jump_diffusion_call(const double& S, const double& K, const double& time_to_maturity, const double& r, const double& sigma, const double& delta, const double& kappa, const double& lambda) {
                                        int it_num = 30;
                                        
                                        double option_price = 0;

                                        for (int k=1; k<it_num; ++k) {
                                            double r_k = r - lambda*(delta - 1) + (k*log(delta))/time_to_maturity;
                                            double sigma_k = sqrt(sigma*sigma + k*delta*delta/time_to_maturity);
                                            option_price += ( exp(-delta*lambda*time_to_maturity) * pow(delta*lambda*time_to_maturity, k)/tgamma(k+1)) * black_scholes_call_price(S, K, r_k, sigma_k, time_to_maturity);
                                        }

                                        return option_price;
                                    }