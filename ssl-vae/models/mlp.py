from typing import List, Optional, Tuple, Type, Union

import torch


class MLP(torch.nn.Module):
    def __init__(self,
                 in_dims: List[int],
                 hidden_dims: List[int],
                 out_dims: List[int],
                 *,
                 hidden_activation: type[torch.nn.Module],
                 out_activations: Optional[List[Optional[Type[torch.nn.Module]]]] = None,
                 zero_init_biases: Optional[bool] = False,
                 init_std: Optional[float] = None,
                 zero_init_output_weights: Optional[List[bool]] = None,
                 zero_init_output_biases: Optional[List[bool]] = None):
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        self.hidden_activation = hidden_activation
        self.out_activations = out_activations
        self.zero_init_output_weights = zero_init_output_weights or [False] * len(self.out_dims)
        self.zero_init_output_biases = zero_init_output_biases or [False] * len(self.out_dims)

        if self.out_activations is not None:
            assert len(self.out_dims) == len(self.out_activations)

        hidden_layers = []
        for i, dim in enumerate(self.hidden_dims):
            linear = torch.nn.Linear(
                sum(self.in_dims) if 0 == i else self.hidden_dims[i-1],
                dim
            )

            if init_std is not None:
                torch.nn.init.normal_(linear.weight, std=init_std)

            if zero_init_biases:
                torch.nn.init.zeros_(linear.bias)
            elif init_std is not None:
                torch.nn.init.normal_(linear.bias, std=init_std)

            hidden_layers.append(linear)
            hidden_layers.append(self.hidden_activation())
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)

        out_modules = []
        for i, (dim, zero_init_weight, zero_init_bias) in enumerate(zip(self.out_dims, self.zero_init_output_weights, self.zero_init_output_biases)):
            linear = torch.nn.Linear(sum(self.in_dims) if 0 == len(self.hidden_dims) else self.hidden_dims[-1], dim)

            if zero_init_weight:
                torch.nn.init.zeros_(linear.weight)
            elif init_std is not None:
                torch.nn.init.normal_(linear.weight, std=init_std)

            if zero_init_bias:
                torch.nn.init.zeros_(linear.bias)
            elif init_std is not None:
                torch.nn.init.normal_(linear.bias, std=init_std)


            layers = [linear]

            if self.out_activations is not None and self.out_activations[i] is not None:
                layers.append(self.out_activations[i]())

            out_modules.append(
                torch.nn.Sequential(*layers) if 1 < len(layers) else layers[0]
            )

        self.out_modules = torch.nn.ModuleList(out_modules)

    def forward(self, *x_: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        if isinstance(x_, tuple):
            x = torch.concat([*x_], dim=1)
        else:
            x = x_
        h = self.hidden_layers(x)
        outputs: Tuple[torch.Tensor] = tuple(m(h) for m in self.out_modules)
        return outputs if len(outputs) > 1 else outputs[0]
