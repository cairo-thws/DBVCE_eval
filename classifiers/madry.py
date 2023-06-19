import torch
import torch.nn as nn
from torchvision import models as torch_models
from .classifier import ClassifierBase
from .paths import MADRY_PATH


class MadryClassifier(ClassifierBase):   
    def __init__(self):
        super().__init__() 

    @property
    def default_path(self):
        return MADRY_PATH
    
    @property
    def robust(self):
        return True
            
    @property
    def name(self) -> str:
        return "MNR-RN50" 
    
    def build_model(self, args=None):
        model_base = torch_models.resnet50
        model_pt = model_base(weights="IMAGENET1K_V1")
        model_pt.eval()
        
        return model_pt
