import os
from easydict import EasyDict as edict

class config :
    def _dataset_cfg( self, dataroot ):
        cfg = edict()
        cfg.dataroot = dataroot

        cfg.templates = edict()
        cfg.templates.mat = '%s/mat/%s_32x32.mat' % ( dataroot, '%s' )
        cfg.templates.pkl = '%s/pkl/%s_32x32.pkl' % ( dataroot, '%s' )
        cfg.templates.hdf5 = '%s/hdf/%s_32x32.hdf5' % ( dataroot, '%s' )

        cfg.type = 'dataset_pkl'

        return cfg

    def _batch_generator_cfg( self ):
        cfg = edict()

        if self.training :
            cfg.batch_size = 128
            cfg.randomize = True
        else :
            cfg.batch_size = 128
            cfg.randomize = False

        return cfg

    def _transform_cfg( self ):
        cfg = edict()

        cfg.num_classes = 10

        return cfg

    def _augmentation_cfg( self ):
        cfg = edict()

        if self.training :
            cfg.tokens = [['randbrightness'], ['randcontrast'], ['randgray', 0.2]]
        else :
            cfg.tokens = []

        return cfg

    def _train_cfg( self ):
        cfg = edict()

        cfg.nepoch = 30
        cfg.lr = 0.01
        cfg.lr_milestones = [15,25]

        return cfg

    def __init__( self, dataroot, training=True ):

        self.training = training
        self._cfg = edict()

        self._cfg.dataset = self._dataset_cfg( dataroot )
        self._cfg.batch_generator = self._batch_generator_cfg()
        self._cfg.transform = self._transform_cfg()
        self._cfg.augmentation = self._augmentation_cfg()
        self._cfg.train = self._train_cfg()

    @property
    def dataset( self ):
        return self._cfg.dataset

    @property
    def batch_generator( self ):
        return self._cfg.batch_generator

    @property
    def transform( self ):
        return self._cfg.transform

    @property
    def augmentation( self ):
        return self._cfg.augmentation

    @property
    def train( self ):
        return self._cfg.train
