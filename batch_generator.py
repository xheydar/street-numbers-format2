import torch
import numpy as np

class batch_generator :
    def __init__( self, cfg, dataset, transform=None ):
        self._cfg = cfg
        self._dataset = dataset
        self._transform = transform
        batch_size = self._cfg.batch_generator.batch_size
        ndata = len(dataset)

        inds = np.arange(ndata)
        
        if self._cfg.batch_generator.randomize :
            np.random.shuffle(inds)

        self._chunks = [ inds[i:i+batch_size] for i in range(0,ndata,batch_size) ]
        self._cur = 0

    def _get_batch( self, idx ):
        chunk = self._chunks[idx]

        inputs = []
        targets = []

        for c in chunk :
            p,l = self._dataset[c]

            b = {}
            b['patch'] = p
            b['label'] = l

            tensor, label = self._transform(b)

            inputs.append( tensor )
            targets.append( label )

        # Ideally this part should be in the transform class
        inputs = torch.stack( inputs, dim=0 )
        targets = torch.from_numpy(np.array(targets))

        targets = torch.nn.functional.one_hot( targets, self._transform.num_classes ).to( torch.float32 )

        return inputs, targets

    def __len__( self ):
        return len(self._chunks)

    def __iter__( self ):
        self._cur = 0
        return self

    def __next__( self ):
        if self._cur >= len(self._chunks) :
            raise StopIteration

        batch = self._get_batch(self._cur)
        self._cur += 1
        
        return batch

    def __getitem__( self, idx ): 
        batch = self._get_batch(idx)
        self._cur += 1
        
        return batch


