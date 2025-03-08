import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, header_hidden_dim=64, num_classes=10):
        """
        Initializes the CIFAR-10 model.

        Args:
            header_hidden_dim (int): The number of hidden units in the first layer of the project header.
            num_classes (int): Number of output classes (default: 10 for CIFAR-10).
        """
        super(CNN, self).__init__()

        # CNN Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=5)  # Input: 3 x 32 x 32, Output: 32 x 28 x 28
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)  # Output: 64 x 24 x 24
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # After pooling: 64 x 12 x 12

        # Compute the flattened feature size after convolutions and pooling.
        # For a 32x32 input: 32 -> (32 - 4 = 28) after conv1, then 28 -> (28 - 4 = 24) after conv2,
        # then pooling gives 24//2 = 12 in each spatial dimension.
        flattened_size = 64 * 12 * 12

        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)

        # Project Header: 2-layer MLP
        self.header_fc1 = nn.Linear(84, header_hidden_dim)
        self.header_fc2 = nn.Linear(header_hidden_dim, num_classes)

    def forward(self, x):
        # CNN Encoder forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Project Header forward pass
        x = self.relu(self.header_fc1(x))
        x = self.header_fc2(x)  # Final logits (no softmax needed as loss functions like CrossEntropyLoss expect raw logits)
        return x