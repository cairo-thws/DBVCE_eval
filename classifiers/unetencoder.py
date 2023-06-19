import torch
import torch.nn as nn
from torchvision import models as torch_models
from .classifier import ClassifierBase
from .paths import UNET_PATH

from guided_diffusion.script_util import (
    classifier_defaults,
    create_classifier,
    args_to_dict,
)

class UnetEncoderClassifier(ClassifierBase):   
    def __init__(self):
        super().__init__() 

    @property
    def default_path(self):
        return UNET_PATH
    
    @property
    def robust(self):
        return True
            
    @property
    def name(self) -> str:
        return "UnetEncoder" 
    
    def build_model(self, args=None):
        model = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        return model
        
    def call(self, x, t=None, preprocessing=True):
        if preprocessing:
            x = self.transform_x(x)
        return self.model(x, t)
