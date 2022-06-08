import numpy as np
import torch
import torchvision.transforms as transforms

class transform :
    def __init__( self, cfg ):
        self._cfg = cfg
        self.num_classes = self._cfg.transform.num_classes

        self._toTensor = transforms.ToTensor()

    def __call__( self, b ):    
        # Augmentation goes here
        tensor = self._toTensor( b['patch'] )
        label = np.int64( b['label'] )

        return tensor, label

