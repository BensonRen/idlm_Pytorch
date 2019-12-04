import torch

if __name__ == '__main__':
    n = torch.zeros([1,10])
    k = torch.ones([1,10])
    print(n)
    print(k)
    dT = torch.div(4 * n, torch.add(torch.pow(torch.add(n, 1), 2), torch.pow(k, 2)))
    print(type(dT.type(torch.float)))



    print("testing slicing from data")
    c = n.data[0,2:5]
    d = n[0,2:5]
    print(type(c))
    print(type(d))


    print("Type of data")
    print(type(n.data))