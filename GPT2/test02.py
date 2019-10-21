import torch
from torch import nn

# a = nn.Embedding(100,10)
# # print(a.weight)
#
# idx = torch.tensor([[1,3,4,5,6],[4,5,3,2,1]])
# print(a(idx))

# print(torch.tril(torch.ones(10, 10)))

a = torch.tensor([0.1, 0.7, 0.2])

for _ in range(10):
    b = torch.multinomial(a, 1)
    print(b)

