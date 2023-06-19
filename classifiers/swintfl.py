import torch
import torch.nn as nn
from torchvision import models as torch_models
from .classifier import ClassifierBase

from simclr import SimCLR

from torchvision.models import swin_b

class SwinTFLClassifier(ClassifierBase):   
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
        return "SwinTFL" 
    
    def load(self, classifier_path=None):
        self._loaded = True
        pass
    
    def build_model(self, args=None):
        model = swin_b(weights='IMAGENET1K_V1')
        
        return model
