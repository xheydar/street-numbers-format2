import numpy as np
import torch
import torchvision.transforms as transforms

class transform :
    def __init__( self, cfg, augment=None ):
        self._cfg = cfg
        self._augment = augment
        self.num_classes = self._cfg.transform.num_classes
        self._toTensor = transforms.ToTensor()

    def __call__( self, b ):    
        # Augmentation goes here
        patch = b['patch']
        if self._augment is not None :
            patch = self._augment( patch )

        tensor = self._toTensor( patch )
        label = np.int64( b['label'] )

        return tensor, label

