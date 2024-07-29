from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers, starting with 1 channel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,1), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1), stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=1920, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=14)

    def forward(self, x):
        # Reshape input data
        x = x.reshape(x.shape[0], 1, 242, 2)

        # Convolutional layers with ReLU activation
        x = self.conv1(x)
        x = x.relu()

        x = self.conv2(x)
        x = x.relu()

        # Flatten for fully connected layers
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers with ReLU activation
        x = self.fc1(x)
        x = x.relu()

        x = self.fc2(x)
        x = x.relu()

        x = self.fc3(x)
        
        # Log softmax for classification
        return x.log_softmax(dim=1)