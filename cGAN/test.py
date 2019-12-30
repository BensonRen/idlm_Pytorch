import torch

if __name__ == '__main__':

    a = torch.rand(4, 0, requires_grad=True)
    print(a)

    b = torch.rand(4, requires_grad=True)

    loss = (a + b.unsqueeze(1)).sum()

    loss.backward()

    print(loss)
