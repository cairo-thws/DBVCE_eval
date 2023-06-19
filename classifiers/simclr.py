import torch
import torch.nn as nn
from torchvision import models as torch_models
from .classifier import ClassifierBase

from simclr import SimCLR



class SIMCLRClassifier(ClassifierBase):   
    def __init__(self):
        super().__init__() 

    @property
    def default_path(self):
        return ""
    
    @property
    def robust(self):
        return True
            
    @property
    def name(self) -> str:
        return "SIMCLR" 
    
    def load(self, classifier_path=None):
        self._loaded = True
        pass
    
    def call(self, x, t=None, preprocessing=True):
        if preprocessing:
            x = self.transform_x(x)
            
        h_i, h_j, z_i, z_j = self.model(x,x)
        return h_i
    
    def build_model(self, args=None):
        encoder = torch_models.resnet50
        encoder = encoder(weights="IMAGENET1K_V1")
        projection_dim = 64
        n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer
        model = SimCLR(encoder, projection_dim, n_features)
        model.eval()
        
        return model
