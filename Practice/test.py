import torch
import numpy as np

x = torch.tensor([[1, 6, 3, 4], [2, 3, 4, 5]])
y = np.arange(5)
print(y)
print(torch.max(x, 0))