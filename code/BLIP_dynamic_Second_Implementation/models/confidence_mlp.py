###########################################################################
#  This module defines a simple Multi-Layer Perceptron (MLP) architecture 
# for predicting confidence scores based on input features. The MLP consists
#  of two hidden layers with ReLU activation and dropout for regularization, 
# followed by a final linear layer that outputs a single confidence score. 
# The input dimension, hidden layer size, and dropout rate can be configured 
# when initializing the model. This confidence MLP can be used in various 
# applications, such as weighting captions based on their predicted quality
# or reliability.
###########################################################################

import torch
import torch.nn as nn

#####################################################################
# The ConfidenceMLP class defines a simple feedforward neural network 
# with two hidden layers and dropout for regularization. The forward 
# method takes an input tensor x and passes it through the network to 
# produce an output confidence score. This model can be trained on 
# features extracted from captions (such as alignment, fluency, 
# and agreement) to predict a confidence score that can be used for 
# weighting captions during training or inference.

# The architecture consists of:
# - An input linear layer that maps the input features to a hidden dimension.
# - A ReLU activation function for non-linearity.
# - A dropout layer to prevent overfitting.
# - Another linear layer followed by ReLU and dropout for the second hidden layer.
# - A final linear layer that outputs a single confidence score.
#######################################################################
class ConfidenceMLP(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)