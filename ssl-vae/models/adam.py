from typing import Union, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, params_t, _use_grad_for_differentiable


class AdaM(Optimizer):
    def __init__(self,
                 params: params_t,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999)):

        defaults = dict(lr=lr, betas=betas, differentiable=False)
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def _init_group(self, group):
        for p in group['params']:
            if p.grad is not None:
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((), dtype=torch.float, device=p.device, requires_grad=False)

                    state['m1'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                    state['m2'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)

                # exp_avgs.append(state['exp_avg'])
                # exp_avg_sqs.append(state['exp_avg_sq'])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']

            self._init_group(group)

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                state['step'] += 1

                step_size = lr  # * torch.sqrt(1.0 - beta2**state['step']) / (1.0 - beta1**state['step'])

                state['m1'] += (1.0-beta1) * (p.grad - state['m1'])
                state['m2'] += (1.0-beta2) * (p.grad**2 - state['m2'])

                diff = state['m1'] / torch.sqrt(state['m2']+1e-8)

                p.add_(-1.0 * step_size * diff)

        return loss
