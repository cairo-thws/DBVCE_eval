import torch
import torch.nn as nn
from torchvision import models as torch_models
from .classifier import ClassifierBase



class AlexNetClassifier(ClassifierBase):    
    def __init__(self):
        super().__init__() 

    @property
    def robust(self):
        return True
            
    @property
    def name(self) -> str:
        return "AlexNet (pretrained)" 
    
    def build_model(self, args=None):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        return model
    
    def default_path(self):
        return None
    
    def load(self, classifier_path):
        self._loaded = True
        # alexnet does not need to load the state dict
        pass



class AlexNetRandomInitClassifier(AlexNetClassifier):    
    def __init__(self):
        super().__init__() 
            
    @property
    def name(self) -> str:
        return "AlexNet (random init)" 
    
    def build_model(self, args=None):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        return model

