import os
import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.desc="This is just a simple base for models to inherit from"
    
    def load(self,source, device='cpu'):
        if os.path.exists(source):
            self.load_state_dict(torch.load(source,map_location=device,weights_only=True))
        else:
            print("Model weights not found")


