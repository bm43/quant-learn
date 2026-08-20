import numpy as np

from quant_learn.vol.garch import Garch

TRUE = (0.02, 0.10, 0.85)


def test_fit_recovers_known_parameters():
    eps, _ = Garch.simulate(*TRUE, n=20000, seed=0)
    fitted = Garch().fit(eps)
    assert np.allclose(fitted, TRUE, atol=0.05)


def test_simulated_variance_is_positive_and_stationary():
    _, var = Garch.simulate(*TRUE, n=5000, seed=1)
    long_run = TRUE[0] / (1 - TRUE[1] - TRUE[2])
    assert (var > 0).all()
    assert abs(var.mean() - long_run) < 0.5 * long_run