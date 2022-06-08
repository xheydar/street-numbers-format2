import platform
import torch
import numpy as np

from config import config
from datasets import datasets
from transform import transform
from batch_generator import batch_generator
import model
from tqdm import tqdm

if platform.system() == "Darwin" :
    dataroot = '/Users/heydar/Work/void/data/datasets/street_numbers/format2'
else :
    dataroot = '/ssd/data/datasets/street_numbers/format2'

class eval_module :
    def __init__( self, dataroot, tag ):
        self._cfg = config( dataroot, training=False )
        self.model_path = 'model_%s.pt' % ( tag )

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

    # Loading the dataset
    def load_dataset( self ):
        self._dataset = datasets[self._cfg.dataset.type](self._cfg, 'test')

    # Creating the batches, adding augmentations and preparing the data for training
    def build_batches( self ):
        t = transform( self._cfg )
        self._batch_gen = batch_generator( self._cfg, self._dataset, transform=t )

    # Loading the model
    def load_model( self ):
        self._model = {}
        self._model['network'] = model.Net( self._cfg )

        model_data = torch.load( self.model_path, map_location='cpu' )
        self._model['network'].load_state_dict( model_data["state_dict"], strict=True )

        self._model['network'] = self._model['network'].to( self.device )
        self._model['network'].eval()

    # Development test function to easily test every is working
    def dev_test( self ):
        inputs, targets = self._batch_gen[10]
        outputs = self._model['network'](inputs)

        print( outputs.argmax(dim=1) )
        print( targets.argmax(dim=1) )

    def eval( self ):

        batch_gen_iter = iter(self._batch_gen)

        actual = []
        pred = []

        for [ inputs, targets ] in tqdm( batch_gen_iter ):

            inputs = inputs.to( self.device )
            targets = targets.to( self.device )

            with torch.no_grad() :
                outputs = self._model['network']( inputs )

                a = targets.argmax( dim=1 ).detach().cpu().numpy()
                p = outputs.argmax( dim=1 ).detach().cpu().numpy()

                actual.append(a)
                pred.append(p)

        actual = np.concatenate( actual )
        pred = np.concatenate( pred )

        print( actual.shape )
        print( pred.shape )

        n_correct = len(np.where( actual == pred )[0])

        print("Acc ", n_correct / len(actual) )


def get_module():
    return eval_module(dataroot, '20220608')


if __name__=="__main__" :
    m = eval_module(dataroot, '20220608')
    m.load_dataset()
    m.build_batches()
    m.load_model()
    m.eval()

