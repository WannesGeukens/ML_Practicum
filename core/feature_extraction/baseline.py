from dabstract.dataprocessor.processing_chain import processing_chain
from dabstract.dataprocessor.processors import *

class baseline(processing_chain):
    def __init__(self):
        super().__init__()
        self.add(Framing(windowsize=0.04, stepsize=0.02, axis=-1))
        self.add(FFT(format='magnitude'))
        self.add(Filterbank(scale='mel', n_bands=40))
        self.add(Logarithm())