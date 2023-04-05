# Sri Harsha Gollamudi 
# Mar 2023
# This code is used to experiment on the parameters of the network implemented. (Task 4)

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from PIL import Image
import cv2
matplotlib.use('TkAgg')

# The network used to train and test the data. It has two convolutional layers, 
# two max pool layers, a drop out layer and a fully connected layer.
class MyNetwork(nn.Module):
    
    # Initializes the layers of the network.
    def __init__(self):
        super(MyNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)
    
    # This method computes a forward pass for the network. It uses relu as the activation function
    #  and a dropout layer for regularization. For the output layer it applies a log softmax activation function.
    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = F.relu(self.maxpool2(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
# This method is used to train the network on the greek letters data.
def train_network(network, optimizer,  epoch, log_interval, train_loader, train_losses, train_counter):

        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                #torch.save(network.state_dict(), 'results/model.pth')
                #torch.save(optimizer.state_dict(), 'results/optimizer.pth')

        return network, train_losses, train_counter

# This method is used to test the network on the testing data using the test_loader.
def test_network(network, test_loader, test_losses):

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return network, test_losses

def gridSearch(epochs, batch_size_train, learning_rate):

    batch_size_test = 1000
    momentum = 0.5
    log_interval = 10

    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_test, shuffle=True)

    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]

    test_network(network, test_loader, test_losses)
    for epoch in range(1, epochs + 1):
        train_network(network, optimizer,  epoch, log_interval, train_loader, train_losses, train_counter)
        test_network(network, test_loader, test_losses)

    return train_losses, train_counter, test_losses, test_counter

    


# This is the main function of the program and it handles all the functions and the network.
def main(argv):
    # handle any command line arguments in argv

    train_losses_list = []
    train_counter_list = []
    test_losses = []
    test_counter = []

    


    return

if __name__ == "__main__":
    main(sys.argv)