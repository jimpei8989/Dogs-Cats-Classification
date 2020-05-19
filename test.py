import importlib

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from Modules.utils import *
from Modules.datahandler import *

SEED = 0x06902029

def main():
    args = parseArguments()

    modelDir = os.path.join('Models', args.name)
    if not os.path.isdir(modelDir):
        print(f'Error: {modelDir} not found')
        exit(0)

    # Set Random Seed
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Load Data
    with EventTimer('Load Data'):
        testX = np.load(os.path.join(args.dataDir, 'testX.npy'))
        print(f'> testX shape: {testX.shape}')
        testDataset = ImageDataset(testX, transform = testTransform)
        testDataloader = DataLoader(testDataset, batch_size = args.batchsize, shuffle = False)

    # Build Model
    Model = importlib.import_module(f'Modules.Models.{args.model}').Model
    model = Model().cuda()

    # Load model weight
    path = os.path.join(modelDir, 'final.wei')
    if not os.path.isfile(path):
        print(f'Error: Weight at {path} not found')
        exit(0)
    else:
        model.load_state_dict(torch.load(path))
        print(f'+ Load model weights from "{path}"')

    with EventTimer('Inference'):
        predictions = []

        for x in testDataloader:
            output = model(x.cuda()).cpu().data.numpy()
            predictions.append(output.reshape(-1))

        predictions = np.concatenate(predictions)

        with open(os.path.join(modelDir, 'prediction.csv'), 'w') as f:
            print('id,label', file = f)
            for i, y in enumerate(predictions):
                print(f'{i + 1},{y:}', file = f)

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('-d', '--datadir', type = str,
            dest = 'dataDir',
            help = 'Data directory')
    parser.add_argument('-n', '--name', type = str,
            dest = 'name',
            help = 'Model name')
    parser.add_argument('-m', '--model', type = str,
            dest = 'model',
            help = 'Model definition')
    parser.add_argument('-b', '--batch_size', type = int,
            dest = 'batchsize', default = 32,
            help = 'batch size (default: 32)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with EventTimer('MAIN'):
        main()

