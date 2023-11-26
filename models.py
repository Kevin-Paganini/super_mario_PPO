import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """_summary_

        Args:
            obs_space (_type_): _description_ 16 x 20 np array
            action_space (_type_): _description_ 5 actions
            num_outputs (_type_): _description_ 5 outputs
            model_config (_type_): _description_ 
            name (_type_): _description_
            
        """
        
        super(TorchModelV2, self).__init__(obs_space, action_space, num_outputs, model_config, name, framework='torch')
        nn.Module.__init__(self)
        # start with 16, 20, 1
        self.conv1 = nn.Conv2d(1, 64, (2, 2), 1, 0)
        # Now we are at 15, 19, 64
        
        
        self.conv2 = nn.Conv2d(64, 64, (2, 2), 1, 0)
        # Now were at 14, 18, 64
        
        self.pool1 = nn.MaxPool2d(2)
        # Now were at 7, 9, 64
        
        self.conv3 = nn.Conv2d(64, 128, (2, 2), 1, 0)
        # Now were at 6, 8, 128
        self.conv4 = nn.Conv2d(128, 128, (2, 2), 1, 0)
        # Now were at 5, 7, 128
        self.conv5 = nn.Conv2d(128, 128, (2, 2), 1, 0)
        # Now were at 4, 6, 128
        
        self.pool2 = nn.MaxPool2d(2)
        # Now were at 2, 3, 128
        
        self.dense1 = nn.Linear(2 * 3 * 128, 256)
        
        self.dense2 = nn.Linear(256, 128)
        
        self.dense3 = nn.Linear(128, 5)
        
        
        
 
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs']
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = F.softmax(x)
        return (x.detach().cpu().numpy(), [])
        
        
        
    def value_function(self):
        return None