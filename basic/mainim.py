import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array


def trick(arg, **kwarg):
    My.handle(*arg, **kwarg)


class My:

    def __init__(self, names):
        self.names = names


    def start(self):

        pool = Pool(processes=2)
        pool.map(self.handle, zip([self] * len(self.names)))

    def handle(self):
        #print(name)
        self.names.append('0')

    def get(self):

        print(self.names)

m = My(names = ['ali', 'veli', '49', '50', 'ayse'])
m.start()
m.get()