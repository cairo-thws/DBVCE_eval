
import torch
import torch.nn as nn
from torchvision import models as torch_models
from .classifier import ClassifierBase

from .ConvNeXt.models.convnext import convnext_large, convnext_base

class ConvNeXtLClassifier(ClassifierBase):    
    def __init__(self):
        super().__init__()
    
    @property
    def robust(self):
        return False

    @property
    def image_size(self):
        return 224

    @property
    def image_size_crop_pct(self):
        return 0.875
    
    @property
    def name(self) -> str:
        return "ConvNeXt-L" 
    
    def build_model(self, args=None):
        model_base = convnext_large(pretrained=True)
        return model_base
    

    def load(self, classifier_path):
        # we already load it with the helper function convnext_large
        self._loaded = True
        pass




class ConvNeXtBaseClassifier(ConvNeXtLClassifier):    
    def __init__(self):
        super().__init__()
    
    @property
    def robust(self):
        return False
            
    @property
    def name(self) -> str:
        return "ConvNeXt-Base" 
    
    def build_model(self, args=None):
        model_base = convnext_base(pretrained=True)
        return model_base
    

    def load(self, classifier_path):
        # we already load it with the helper function convnext_large
        pass
