import pickle
from PIL import Image

class dataset_pkl :
    def __init__( self, cfg, part ):
        self._cfg = cfg

        with open(self._cfg.dataset.templates.pkl % (part), 'rb') as ff :
            data = pickle.load(ff)

        self._data = data['X']
        self._labels = data['y']

        self._ndata = len(self._data)

    def __len__( self ):
        return self._ndata

    def __getitem__( self, idx ):
        p = Image.fromarray(self._data[idx])
        l = self._labels[idx]

        return p,l
