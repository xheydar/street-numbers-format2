import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

class trainer :
    def __init__( self ):
        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

    def _set_device( self, blobs ):
        if type(blobs) == list :
            for idx, b in enumerate( blobs ) :
                blobs[idx] = self._set_device( b )
        elif type(blobs) == dict :
            for key, b in blobs.items() :
                blobs[key] = self._set_device(b)
        elif type(blobs) != int :
            blobs = blobs.to( self.device )
        return blobs

    def train( self ):
        
        nepoch = self._cfg.train.nepoch
        lr = self._cfg.train.lr
        lr_milestones = self._cfg.train.lr_milestones

        optimizer = optim.SGD( self._model['network'].parameters(),
                               lr=lr,
                               momentum=0.9,
                               weight_decay=1e-4 )


        scheduler = optim.lr_scheduler.MultiStepLR( optimizer, milestones=lr_milestones, 
                                                    gamma=0.1 )


        train_loader = DataLoader( self._batch_gen, batch_size=None )
        
        for epoch_idx in range( nepoch ):
            print("Epoch : %d - %d" % (epoch_idx+1, nepoch))

            train_iter = iter( train_loader )

            losses = []

            for [ inputs, targets ] in tqdm( train_iter ):
                inputs = self._set_device( inputs )
                targets = self._set_device( targets )

                optimizer.zero_grad()

                outputs = self._model['network']( inputs )
                loss = self._model['loss']( outputs, targets )

                losses.append( float(loss) )

                loss.backward()
                optimizer.step()

            print('Average Loss : ', np.mean(losses))

            scheduler.step()

        print( self.model_path )
        torch.save({'state_dict':self.model['network'].state_dict()}, self.model_path)

