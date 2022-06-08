import numpy as np
from PIL import ImageFilter, ImageEnhance

class randbrightness :
    def __init__( self ):
        pass

    def __call__( self, image ):
        factor = np.random.uniform(0.5,1.5)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance( factor )

class randcontrast :
    def __init__( self ):
        pass

    def __call__( self, image ):
        factor = np.random.uniform(0.5,1.5)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance( factor )

class randgray :
    def __init__( self, prob ):
        self._prob = prob

    def __call__( self, image ):
        p = np.random.rand()
        if p < self._prob :
            image = image.convert('LA').convert('RGB')
        return image

class randblur :
    def __init__( self, radius, prob ):
        self._prob = prob
        self._radius = radius

    def __call__( self, image ):
        p = np.random.rand()
        if p < self._prob :
            image = image.filter( ImageFilter.GaussianBlur(radius=self._radius) )
        return image

class augmentation :
    def __init__( self, cfg ):

        self._functions = {}
        self._functions['randbrightness'] = randbrightness
        self._functions['randcontrast'] = randcontrast
        self._functions['randgray'] = randgray
        self._functions['randblur'] = randblur

        self._ops = []

        for t in cfg.augmentation.tokens :
            self._ops.append( self._functions[t[0]](*t[1:]) )


    def __call__( self, image ):
        
        for op in self._ops :
            image = op(image)

        return image
