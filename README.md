# quant-learn

Volatility and regime-switching models written from scratch while learning
quantitative finance. Written to understand the mechanics, not as a library.

## Contents

**`src/quant_learn/vol`**
- `garch.py` : GARCH(1,1) simulation and quasi-maximum-likelihood fitting (https://math.berkeley.edu/~btw/thesis4.pdf)
- `log_normal_mixture.py` : lognormal mixture density (https://quant.opengamma.io/Mixed_Log-Normal-Volatility-Model.pdf)

**`src/quant_learn/regime`**
- `markov_switch.py` : Markov regime-switching model (https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf)
- `hidden_markov.py` : hidden Markov model, forward-backward (https://web.stanford.edu/~jurafsky/slp3/A.pdf)

**`cpp/imp_vol_solver`**
Implied volatility solver in C++17. Three root finders share one Black-Scholes
functor: interval bisection, Brent's method, and Newton-Raphson using vega.

## Run

    pip install -e ".[dev]"
    pytest

    cd cpp/imp_vol_solver
    make test     # solvers recover a known vol from a price they generated
    make && ./solver

## Status

Learning repo, finished and no longer developed.