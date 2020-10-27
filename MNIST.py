from torchvision import datasets, transforms
import torch
from torch import optim, nn
import model

# normalize the data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


mod = model.Network(784, 10, [64])
optimizer = optim.Adam(mod.parameters(), lr=0.003)
criterion = nn.NLLLoss()
model.train(mod, trainloader, testloader, criterion, optimizer)
print('meow')