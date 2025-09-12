from typing import List, Tuple
import numpy as np


def fedavg_weighted(updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    total_examples = sum(n for _, n in updates)
    weighted = None
    for params, n in updates:
        scale = n / total_examples
        if weighted is None:
            weighted = [p * scale for p in params]
        else:
            for i in range(len(params)):
                weighted[i] += params[i] * scale
    return weighted if weighted is not None else []
