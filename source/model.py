import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim):
        ''' Defines layers of a neural network.
        
        param input_dim: 
            - Number of input features
        param hidden_dim_list:  
            - List of hidden layer dimensions
        param output_dim: 
            - Number of output
        
        '''
        assert len(hidden_dim_list) > 0 and all([isinstance(i, int) for i in hidden_dim_list])
        super(DynamicNet, self).__init__()
        full_dim_list = [input_dim] + hidden_dim_list + [output_dim]
        self.full_dim_list = full_dim_list
        self.fc_dict = {}
        for i in range(len(full_dim_list) - 1):
            fc = nn.Linear(full_dim_list[i], full_dim_list[i + 1])
            setattr(self, f'fc{i + 1}', fc)
            self.fc_dict[i + 1] = f'fc{i + 1}'
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        ''' Feedforward behavior of the net
        
        param x: 
            - A batch of input features
        return:
            - A single, sigmoid activated value
        
        '''
        for i in range(len(self.fc_dict)):
            fc = getattr(self, self.fc_dict[i + 1])
            x = fc(x)
            if i + 1 != len(self.fc_dict):
                x = self.dropout(x)
        return x