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


class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Hardtanh(),
            nn.Linear(128, 64),
            nn.Hardtanh(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# address key mismatch issue
state_dict = torch.load('pretrained/hardtanh_model.pth')
new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
model = MNIST()
model.load_state_dict(new_state_dict) 

## Step 2: Prepare dataset as usual
N = 1 # only one image in problem4
n_classes = 10
image, true_label = torch.load('data/data1.pth')
image = image[:N].view(N,1,28,28) # add batch size before
true_label = torch.tensor([true_label])

## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph,
# and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
print('Running on', image.device)
# Visualize the lirpa_model
# Visualization file is saved as "bounded_mnist_model.png" or "bounded_mnist_model.dot"
lirpa_model.visualize("bounded_mnist_model")
print()

## Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.01
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = lirpa_model(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()

# no comparison
method = 'backward (CROWN)'

print('Bounding method:', method)
lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
for i in range(N):
    print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
    for j in range(n_classes):
        indicator = '(ground-truth)' if j == true_label[i] else ''
        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
print()

# There are many bound coefficients during CROWN bound calculation; here we are interested in the linear bounds
# of the output layer, with respect to the input layer (the image).
required_A = defaultdict(set)
required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])

lb, ub, A_dict = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], return_A=True, needed_A_dict=required_A)
lower_A, lower_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
upper_A, upper_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA'], A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
print(f'lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}')
print(f'lower bound linear coefficients norm (smaller is better): {lower_A.norm()}')
print(f'lower bound bias term size (batch, output_dim): {list(lower_bias.size())}')
print(f'lower bound bias term sum (larger is better): {lower_bias.sum()}')
print(f'upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}')
print(f'upper bound linear coefficients norm (smaller is better): {upper_A.norm()}')
print(f'upper bound bias term size (batch, output_dim): {list(upper_bias.size())}')
print(f'upper bound bias term sum (smaller is better): {upper_bias.sum()}')
print(f'These linear lower and upper bounds are valid everywhere within the perturbation radii.\n')

## An example for computing margin bounds.
# In compute_bounds() function you can pass in a specification matrix C, which is a final linear matrix applied to the last layer NN output.
# For example, if you are interested in the margin between the groundtruth class and another class, you can use C to specify the margin.
# This generally yields tighter bounds.
# Here we compute the margin between groundtruth class and groundtruth class + 1.
# If you have more than 1 specifications per batch element, you can expand the second dimension of C (it is 1 here for demonstration).
lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
C = torch.zeros(size=(N, 1, n_classes), device=image.device)
groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
target_label = (groundtruth + 1) % n_classes
C.scatter_(dim=2, index=groundtruth, value=1.0)
C.scatter_(dim=2, index=target_label, value=-1.0)
print('Specification matrix:\n', C)

lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
for i in range(N):
    print('Image {} top-1 prediction {} ground-truth {}'.format(i, label[i], true_label[i]))
    print('margin bounds: {l:8.3f} <= f_{j}(x_0+delta) - f_{target}(x_0+delta) <= {u:8.3f}'.format(
        j=true_label[i], target=(true_label[i] + 1) % n_classes, l=lb[i][0].item(), u=ub[i][0].item()))
print()
