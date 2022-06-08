import scipy.io
import pickle
from config import config
import numpy as np

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

        data = {}
        data['X'] = X
        data['y'] = y

        with open( pkl_path, 'wb' ) as ff :
            pickle.dump( data, ff )

