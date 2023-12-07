from typing import Dict, Mapping, Union

import torch


class EMA:
    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.dictionary = {}

    def update(self, d: Mapping[str, Union[float, torch.Tensor]]) -> Dict[str, float]:
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if key in self.dictionary:
                self.dictionary[key] = self.decay * self.dictionary[key] + (1.0-self.decay) * value
            else:
                self.dictionary[key] = value
        return self.dictionary
