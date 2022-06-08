
# 1. Preparing the dataset

The dataset provided in matlab format. The first step in our analysis is to convert 
this dataset into a format that is easier for python to work with. We are going to 
use the scripts mat2pkl.py and mat2file.py to covert the mat file into a more proper
format.

 - mat2pkl.py : In this case, we will just save the data within the mat file as a 
   pickle file. This will mostly be used for training.
 - mat2file.py : To show case "online" data loading, this script will store that data
   as individual jpg images.

Note : To run this script, please set the dataroot variable withing these files 
accordingly.

# 2. Configurations

Most of the behaviour of the provided code is controlled via the variables defined in
the config.py file.

# 3. Model

I'm using a rather simple model with Efficientnet-B0 as the backbone and a simple 
classifier head for solving this problem.

# 3. Training

To train the model, run train.py ( dataroot variable should be adjusted ). A trained
model is provided in the repository. This model is trained without data augmentation.

# 4. Evaluation

To evaluate a trained model, run eval.py. I have not optimized this code for the best 
accuracy. The accuracy model provided in the repository [ model_20220608.pt ] on the
test set is 0.8774.

# 6. Dataset / Dataloader

Two dataset classes are provided in the dataset/ folder. They can be used interchangeably
and we can choose in config.py which dataset class we want to use. [ Lines 14 and 15 ]

 - dataset_pkl : The whole data is loaded from the pickle file
 - dataset_files : Only a reference to the images and the labels are loaded. Each image is
   loaded when it is called when we are putting the data batch together.

# 7. Batch generator

This is very standard batch generator that randomizes the samples found in the 
dataset and prepares the batches.  

# 8. Transform and Augmentation

Transform converts the input data to a format that can be used by pytorch for training or
evaluation purposes. The augmentation class contains a number of simple adjustment that 
are applied randomly to the images. The current code is designed in a way that the augmentation
is only called during the training. We can control from config.py which augmentations we 
want to use.
