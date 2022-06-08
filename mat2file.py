import os
import scipy.io
import pickle
from config import config
import numpy as np
from PIL import Image
from tqdm import tqdm

dataroot = '/Users/heydar/Work/void/data/datasets/street_numbers/format2'

"""
    We will organize the data in the dataroot directory as the following :

        dataroot -
            - mat : Original Matlab mat files
            - pkl : Converted pickle files
            - files : Raw files 
"""


if __name__=="__main__" :
    cfg = config(dataroot)

    datasets = ['train', 'test']

    files_root = os.path.join( dataroot, 'files' )

    for d in datasets :
        mat_path = cfg.dataset.templates.mat % (d)
        pkl_path = cfg.dataset.templates.pkl % (d)
        
        #
        # Loading the matfile
        #
        
        mat_data = scipy.io.loadmat( mat_path )

        X = mat_data['X']
        
        #
        # Transposing the data to have N as the first element
        #

        X = X.transpose([3,0,1,2])

        #
        # Subtracting 1 from the labels to map them to 0 to 9 instead of 1 to 10
        #

        y = mat_data['y'] - 1
        y = y.ravel()


        if not os.path.exists( os.path.join( files_root, d ) ):
            os.mkdir( os.path.join( files_root, d ) )
            os.mkdir( os.path.join( files_root, d, 'images' ) )

        ndata = len(y)

        imlist = []

        for idx in tqdm( range(ndata) ): 
            im_path = '%s/images/image_%d.jpg' % ( d, idx )
            img_pil = Image.fromarray( X[idx] )

            imdata = {}
            imdata['path'] = im_path
            imdata['label'] = y[idx]

            p = os.path.join( files_root, im_path )

            img_pil.save( p )

            imlist.append( imdata )

        with open( os.path.join( files_root, d, 'imlist.pkl' ), 'wb' ) as ff :
            pickle.dump( imlist, ff )




        #data = {}
        #data['X'] = X
        #data['y'] = y

        #with open( pkl_path, 'wb' ) as ff :
        #    pickle.dump( data, ff )

