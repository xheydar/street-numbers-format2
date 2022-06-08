
# 1. Preparing the dataset

The dataset provided in matlab format. The first step in our analysis is to convert 
this dataset into a format that is easier for python to work wtih. We are going to 
use the scripts mat2pkl.py and mat2file.py to covert the mat file into a more proper
format.

 - mat2pkl.py : In this case, we will just save the data within the mat file as a 
   pickle file. This will mostly be used for training.
 - mat2file.py : To show case "online" data loading, this script will store that data
   as individual jpg images.

Note : To run this script, please set the dataroot variable withing these files 
accordingly.
