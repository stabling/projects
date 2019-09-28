import torch


a = torch.tensor([1., 3, 10, 5, 6])
b = torch.argmax(a)
print(b)
