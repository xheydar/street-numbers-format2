import os
import pickle
from PIL import Image

class dataset_files :
    def __init__( self, cfg, part ):
        self._cfg = cfg

        # Loading image list

        with open( self._cfg.dataset.templates.files % (part), 'rb' ) as ff :
            imlist = pickle.load(ff)

        self._imlist = imlist
        self._ndata = len(self._imlist)
        
    def __len__( self ):
        return self._ndata

    def __getitem__( self, idx ):
        #
        # Fetching the label from the imlist
        #

        l = self._imlist[idx]['label']
        im_path = os.path.join( self._cfg.dataset.dataroot, 
                                'files', self._imlist[idx]['path'] )

        #
        # Loading the image from the file
        #
        p = Image.open( im_path )

        return p,l
