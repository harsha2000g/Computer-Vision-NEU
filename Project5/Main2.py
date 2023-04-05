# Sri Harsha Gollamudi 
# Mar 2023
# This code is used to examine the network and analyze how it processes the data. (Task 2)

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

# This is the main function of the program and it handles all the functions and the network.
def main(argv):
    # handle any command line arguments in argv

    # Loading the saved network
    network = MyNetwork()
    savedModel = torch.load('results/model.pth')
    network.load_state_dict(savedModel)

    print("The network is: ", network)

    print("Shape of the first layer weights: ", network.conv1.weight.shape)

    weights = network.conv1.weight
    for i in range(10):
        print("Shape of filter ", i, ": ", weights[i, 0].shape)
        print("Weights of filter ", i, ": ", weights[i, 0])


    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        if i < 10:
            img = ax.imshow(weights[i, 0].detach().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Filter {}".format(i+1))
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=1, shuffle=True)

    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    image = example_data

    with torch.no_grad():
        img = example_data[0][0].numpy()
        fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12,5))
        filtered_images = []
        for i in range(5):
            filter = weights[i, 0].detach().numpy()
            filtered_image = cv2.filter2D(img, -1, filter)
            filtered_images.append(filtered_image)
            axes[i,1].imshow(filtered_image, cmap = "gray")
            axes[i,1].set_xticks([])
            axes[i,1].set_yticks([])
            axes[i,0].imshow(filter, cmap = "gray")
            axes[i,0].set_xticks([])
            axes[i,0].set_yticks([])
        for i in range(5, 10):
            x = i - 5
            filter = weights[i, 0].detach().numpy()
            filtered_image = cv2.filter2D(img, -1, filter)
            filtered_images.append(filtered_image)
            axes[x,3].imshow(filtered_image, cmap = "gray")
            axes[x,3].set_xticks([])
            axes[x,3].set_yticks([])
            axes[x,2].imshow(filter, cmap = "gray")
            axes[x,2].set_xticks([])
            axes[x,2].set_yticks([])
        plt.tight_layout()
        plt.show()


    return

if __name__ == "__main__":
    main(sys.argv)