'''
RTN, round-to-nearest, as a Quantizer. Adrian's original is archive/_rtn.py.

Each row of W is cut into groups of group_size weights. A group's grid has 2^bits evenly
spaced values from its min to its max, and every weight moves to the closest one
(round_to_groups in base.py). Only the weights are looked at, so there is no calibration.
'''
import torch

from qlrl.quant.base import Quantizer


class RTN(Quantizer):
    def quantize_matrix(self, name: str, matrix: torch.Tensor) -> torch.Tensor:
        """Return `matrix` [out, in] on its RTN grid."""
        return self.round(matrix)
