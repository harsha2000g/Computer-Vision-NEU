# Sri Harsha Gollamudi
# This code is 

# import statements
import sys
import torch
import torch.nn as nn
import torchvision

# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
         pass
    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        return x

# useful functions with a comment for each function
def train_network( arguments ):
        
        return

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_test, shuffle=True)

    

    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)