# Sri Harsha Gollamudi 
# Mar 2023
# This code is used to test the saved network. 

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from torchviz import make_dot
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

# This function loads the custom images that are handwritten and resizes them to match the input dimensions (28 x 28)    
def loadCustomImages():
    customImages = []
    for i in range(10):
        image = Image.open(f'Custom Digits/{i}.jpg').convert('L')
        image = torchvision.transforms.Resize((28, 28))(image)
        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Normalize((0.1307,), (0.3081,))(image)
        customImages.append(image)
    return customImages

# This is the main function of the program and it handles all the functions and the network.
def main(argv):
    # handle any command line arguments in argv

    # Loading the saved network
    network = MyNetwork()
    savedModel = torch.load('results/model.pth')
    network.load_state_dict(savedModel)

    network.eval()

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
    batch_size=1, shuffle=True)

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    fig.subplots_adjust(hspace = .5, wspace=.001)

    for i, (img, label) in enumerate(test_loader):
        if i >= 9:
            break

        output = network(img)

        print("Image ", i+1, ": ", output.detach().numpy()[0].round(2))
        print("Max/Predicted Class: ", output.argmax().numpy())

        print("Correct Class: ", label.item())

        img = img.squeeze()
        img = img.detach().numpy()
        axs[i//3, i%3].imshow(img, cmap='gray')
        axs[i//3, i%3].set_title("Pred: {}".format(output.argmax().numpy()))
        axs[i//3, i%3].axis('off')

    plt.show()

    customImages = loadCustomImages()

    print("\nThe Custom Data output is\n")

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    fig.subplots_adjust(hspace = .5, wspace=.001)

    for i, img in enumerate(customImages):

        output = network(img.unsqueeze(0))

        print("Custom Image ", i+1, ": ", output.detach().numpy()[0].round(2))
        print("Max/Predicted Class: ", output.argmax().numpy())

        if i < 9:
            img = img.squeeze()
            img = img.detach().numpy()
            axs[i//3, i%3].imshow(img, cmap='gray')
            axs[i//3, i%3].set_title("Pred: {}".format(output.argmax().numpy()))
            axs[i//3, i%3].axis('off')
    
    plt.show()

    # vis_graph = make_dot(network(img.unsqueeze(0)), params=dict(network.named_parameters()))
    # vis_graph.render('Network Diagram', format='png')

    return

if __name__ == "__main__":
    main(sys.argv)