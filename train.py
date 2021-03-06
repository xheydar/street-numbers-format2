import platform
import torch

from config import config
from datasets import datasets
from augmentation import augmentation
from transform import transform
from batch_generator import batch_generator
import model
from trainer import trainer

if platform.system() == "Darwin" :
    dataroot = '/Users/heydar/Work/void/data/datasets/street_numbers/format2'
else :
    dataroot = '/ssd/data/datasets/street_numbers/format2'

class train_module( trainer ) :
    def __init__( self, dataroot, tag ):
        super().__init__()

        self._cfg = config( dataroot )
        self.model_path = 'model_%s.pt' % ( tag )

    # Loading the dataset
    def load_dataset( self ):
        self._dataset = datasets[self._cfg.dataset.type](self._cfg, 'train')

    # Creating the batches, adding augmentations and preparing the data for training
    def build_batches( self ):

        #
        # Augmentation module to randomly modify the input images
        #

        augment = augmentation( self._cfg ) 


        #
        # Transform will prepare the loaded data to something more suitable for the
        # neural network
        #

        t = transform( self._cfg, augment=augment )
        self._batch_gen = batch_generator( self._cfg, self._dataset, transform=t )

    # Loading the model
    def load_model( self ):
        self._model = {}
        self._model['network'] = model.Net( self._cfg ).to( self.device )
        self._model['loss'] = model.Loss().to( self.device )

        #model_data = torch.load( self.model_path, map_location='cpu' )
        #self._model['network'].load_state_dict( model_data["state_dict"], strict=True )

    # Development test function to easily test every is working
    def dev_test( self ):
        inputs, targets = self._batch_gen[10]
        outputs = self._model['network'](inputs)
        loss = self._model['loss'](outputs, targets)

        print( outputs.argmax(dim=1) )
        print( targets.argmax(dim=1) )

        print( loss )

def get_module():
    return train_module(dataroot, '20220608')


if __name__=="__main__" :
    m = train_module(dataroot, '20220608')
    m.load_dataset()
    m.build_batches()
    m.load_model()
    m.train()

