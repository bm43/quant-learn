"""
hamilton's regime switching model

https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf

  https://personal.eur.nl/kole/rsexample.pdf

  https://www.stata.com/features/overview/markov-switching-models/

  https://homepage.ntu.edu.tw/~ckuan/pdf/Lec-Markov_note.pdf
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import 

@dataclass
class MarkovSwitchModel:
  
  def _sigmoid(self, x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))