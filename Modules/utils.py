import os, sys, time
import pickle
from argparse import ArgumentParser

import numpy as np
import torch

_LightGray = '\x1b[38;5;251m'
_Bold = '\x1b[1m'
_Underline = '\x1b[4m'
_Orange = '\x1b[38;5;215m'
_SkyBlue = '\x1b[38;5;38m'
_Reset = '\x1b[0m'

class EventTimer():
    def __init__(self, name = '', verbose = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(_LightGray + '------------------ Begin "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + '" ------------------' + _Reset, file = sys.stderr)
        self.beginTimestamp = time.time()
        return self

    def __exit__(self, type, value, traceback):
        elapsedTime = time.time() - self.beginTimestamp
        if self.verbose:
            print(_LightGray + '------------------ End "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + ' (Elapsed ' + _Orange + f'{elapsedTime:.4f}' + _Reset + 's)" ------------------' + _Reset + '\n', file = sys.stderr)

    def gettime(self):
        return time.time() - self.beginTimestamp

def pickleSave(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickleLoad(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def printEpoch(epoch, numEpochs, elapsedTime, trainLoss, trainAcc, validLoss, validAcc):
    _Train = '\x1b[38;5;210m'
    _Valid = '\x1b[38;5;220m'
    print(f'Epoch {epoch:3d} / {numEpochs:3d} [{elapsedTime:.2f}s] ', end = '')
    print(_Train + f'Training   ~> CE : {trainLoss:.4f} | ACC : {trainAcc:.4f}' + _Reset, end = ' ; ')
    #  print('                         ', end = '')
    print(_Valid + f'Validation ~> CE : {validLoss:.4f} | ACC : {validAcc:.4f}' + _Reset)

def Accuracy(pred, truth):
    return np.mean(np.round(pred).reshape(-1) == truth.reshape(-1))

