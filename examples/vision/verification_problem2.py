"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten

class simple_model(nn.Module):

    def __init__(self):
        super().__init__()
        # Weights of linear layers.
        self.w1 = torch.tensor([[1., -1.], [2., -2.]])
        self.b1 = torch.tensor([1., 1.])
        self.w2 = torch.tensor([[1., -1.], [2., -2.]])
        self.b2 = torch.tensor([2., 2.])
        self.w3 = torch.tensor([[-1., 1.]])

    def forward(self, x):
        # First linear layer
        z1 = x.matmul(self.w1.t()) + self.b1
        hz1 = nn.functional.relu(z1)

        # Second linear layer
        z2 = hz1.matmul(self.w2.t()) + self.b2
        hz2 = nn.functional.relu(z2)

        # Skip connection and final output
        y = (hz2 + z1).matmul(self.w3.t())
        return y


model = simple_model()

# Input x.
x = torch.tensor([[0., 0.]])
# Lowe and upper bounds of x.
lower = torch.tensor([[-1., -1.]])
upper = torch.tensor([[1., 1.]])

lirpa_model = BoundedModule(model, torch.empty_like(x))
pred = lirpa_model(x)
print(f'Model prediction: {pred.item()}')

norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
bounded_x = BoundedTensor(x, ptb)

# Compute bounds.
# IBP method
lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='IBP')
print(f'IBP bounds: lower={lb.item()}, upper={ub.item()}')

intermediate_bounds = lirpa_model.save_intermediate()
print("\nIntermediate Layer Bounds (IBP):")
for layer, (lb, ub) in intermediate_bounds.items():
    print(f"Layer {layer}: Lower Bound = {lb.flatten().tolist()}, Upper Bound = {ub.flatten().tolist()}")

# CROWN method
lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
print(f'\nCROWN bounds: lower={lb.item()}, upper={ub.item()}')

intermediate_bounds = lirpa_model.save_intermediate()
print("\nIntermediate Layer Bounds (CROWN):")
for layer, (lb, ub) in intermediate_bounds.items():
    print(f"Layer {layer}: Lower Bound = {lb.flatten().tolist()}, Upper Bound = {ub.flatten().tolist()}")

# # Getting the linear bound coefficients (A matrix) for each layer.
required_A = defaultdict(set)
for layer in lirpa_model._modules.keys():
    required_A[layer].add(lirpa_model.input_name[0])


lb, ub, A = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN', return_A=True, needed_A_dict=required_A)
print('\nCROWN linear (symbolic) bounds for each layer: lA x + lbias <= f(x) <= uA x + ubias, where')
for layer, a_dict in A.items():
    print(f"Layer {layer}:")
    for input_name, a_matrix in a_dict.items():
        print(f"  Input {input_name}: A matrix = {a_matrix}")