import platform

from config import config
from datasets import datasets
from transform import transform
from batch_generator import batch_generator
import model
from trainer import trainer

if platform.system() == "Darwin" :
    dataroot = '/Users/heydar/Work/void/data/datasets/street_numbers/format2'
else :
    dataroot = '/ssd/'

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
        t = transform( self._cfg )
        self._batch_gen = batch_generator( self._cfg, self._dataset, transform=t )

    # Loading the model
    def load_model( self ):
        self._model = {}
        self._model['network'] = model.Net( self._cfg ).to( self.device )
        self._model['loss'] = model.Loss().to( self.device )

    # Development test function to easily test every is working
    def dev_test( self ):
        inputs, targets = self._batch_gen[10]
        outputs = self._model['network'](inputs)
        loss = self._model['loss'](outputs, targets)

def get_module():
    return train_module(dataroot, '20220608')


if __name__=="__main__" :
    m = train_module(dataroot, '20220608')
    m.load_dataset()
    m.build_batches()
    m.load_model()
    m.train()

