from typing import List

from numpy.core.multiarray import ndarray


class ModelInput:
    def __init__(self, embedding_weights: List[ndarray] = None, x: ndarray = None, y: List = None,
                 mode: str = 'embedding') -> None:
        self.embedding_weights: List[ndarray] = embedding_weights
        self.x: ndarray = x
        self.y: List = y
        self.mode: str = mode
