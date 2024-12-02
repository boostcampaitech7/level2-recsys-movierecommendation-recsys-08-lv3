import numpy as np

class EASE:
    def __init__(self, _lambda):
        """
        Initialize the EASE class with a regularization parameter.

        Parameters:
        - _lambda: Regularization parameter for the diagonal elements.
        """
        self.B = None
        self._lambda = _lambda