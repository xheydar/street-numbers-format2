from torch import nn
from timm import create_model

class Net( nn.Module ):
    def __init__( self, cfg ):
        super().__init__()

        #
        # Using a simple efficientnet backbone for feature extraction
        #

        self.backbone = create_model( model_name='efficientnet_b0',
                                      features_only=True,
                                      pretrained=False )

        nclasses = cfg.transform.num_classes

        #
        # Building a simple head for classification
        #

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, 64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward( self, x ):
        x = self.backbone(x)[-1]
        x = self.head(x)
        return x

class Loss( nn.Module ):
    def __init__( self ):
        super().__init__()

        #
        # The loss
        #

        self.loss = nn.CrossEntropyLoss()

    def forward( self, outputs, targets ):
        return self.loss( outputs, targets )
