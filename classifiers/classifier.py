from abc import ABC, abstractmethod
from guided_diffusion import dist_util
from torchvision import transforms
import torch
import math, gc
from PIL import Image

class ClassifierBase(ABC):    
    def __init__(self):
        super().__init__()
        self._model = None
        self._loaded = False
    
    def load(self, classifier_path=None):
        if classifier_path is None or classifier_path == "":
            classifier_path = self.default_path
        self._model.load_state_dict(
            dist_util.load_state_dict(classifier_path, map_location="cpu")
        )
        self._loaded = True
        
    def perturbate(self, ratio):
        assert self._loaded, "Classifier not loaded"
        assert ratio >= 0 and ratio <= 1, "Ratio must be between 0 and 1"
        # grab all the parameters of the model and get the mean and std
        # of the parameters
        
        if ratio == 0:
            return
        
        # bring all parameters in one long vector
        p  = torch.cat([p.flatten() for p in self._model.parameters()])
        std = torch.std(p)
        
        # ratio is the percentage of the std to add to the mean
        # so we add the mean + ratio*std to each parameter
        for p in self._model.parameters():
            # pull from normal distribution
            # TODO: maybe we have to add mean too?
            p.data += torch.randn(p.data.size())*std*ratio
         
        
        
    def init_model(self, args):
        self._model = self.build_model(args)

    def transform_x(self,x):
        img_size = self.image_size
        crop_pct = self.image_size_crop_pct
        
        size = int(math.floor(img_size / crop_pct))
        return transforms.Resize(size, interpolation=Image.BICUBIC, antialias=False)(x)

    @property
    def image_size(self):
        return 256
    
    @property
    def image_size_crop_pct(self):
        return 1

    @property
    def default_path(self):
        raise NotImplementedError
        
    @property
    def model(self):
        return self._model
        
    @abstractmethod    
    def build_model(self, args=None):
        raise NotImplementedError
        
        
    def call(self, x, t=None, preprocessing=True):
        if preprocessing:
            x = self.transform_x(x)
        return self.model(x)
    
    @property
    @abstractmethod
    def robust(self) -> bool:
        raise NotImplementedError     
    
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError 
    
    # define tostring; if robust, add robust to name
    def __str__(self):
        return self.name + " "+ ("(robust)" if self.robust else "(non-robust)")
    
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        print(f"Converting {self.name} to fp16")
        self._model = self._model.half()

    def __delete__(self):
        print(f"__delete__: {self.name}")
        if self._model is not None:
            # move model to cpu
            self._model.to("cpu")
            # delete model
            del self._model
            self._model = None
            # and clear memory
            gc.collect()
            torch.cuda.empty_cache()
     

        

    
    
    