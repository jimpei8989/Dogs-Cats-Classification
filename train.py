import importlib

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Modules.utils import *
from Modules.datahandler import *

SEED = 0x06902029

def main():
    args = parseArguments()

    modelDir = os.path.join('Models', args.name)
    try:
        os.mkdir(modelDir)
        print(f'+ Create model directory {modelDir}')
    except FileExistsError:
        pass

    # Set Random Seed
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Load Data
    with EventTimer('Load Data'):
        trainX = np.load(os.path.join(args.dataDir, 'trainX.npy'))
        trainY = np.load(os.path.join(args.dataDir, 'trainY.npy'))
        validX = np.load(os.path.join(args.dataDir, 'validX.npy'))
        validY = np.load(os.path.join(args.dataDir, 'validY.npy'))
        
        print(f'> trainX shape: {trainX.shape} ({trainX.dtype})')
        print(f'> trainY shape: {trainY.shape} ({trainY.dtype})')
        print(f'> validX shape: {validX.shape}')
        print(f'> validY shape: {validY.shape}')

        trainDataset = ImageDataset(trainX, trainY, trainTransform)
        validDataset = ImageDataset(validX, validY, testTransform)

        trainDataloader = DataLoader(trainDataset, batch_size = args.batchsize, shuffle = True)
        validDataloader = DataLoader(validDataset, batch_size = args.batchsize, shuffle = False)

    # Build Model
    Model = importlib.import_module(f'Modules.Models.{args.model}').Model
    model = Model().cuda()
    model.summary()
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr = args.lr)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor = 0.5, min_lr = 1e-6,
                                  threshold = 1e-3,
                                  patience = 7, cooldown = 3,
                                  verbose = True
                                  )

    startEpoch = 0
    # Load model weight
    try:
        path = os.path.join(modelDir, 'final.wei')
        model.load_state_dict(torch.load(path))
        print('+ Load model weights')
    except:
        print('+ A brand new model')

    numEpochs = args.epochs

    with EventTimer('Train Model'):
        history = []
        for epoch in range(1, numEpochs + 1):
            trainLosses, validLosses = [], []

            with EventTimer(verbose = False) as et:
                # Train
                model.train()
                for x, y in trainDataloader:
                    optimizer.zero_grad()
                    output = model(x.cuda()).cpu()

                    loss = criterion(output.double(), y.double())
                    loss.backward()
                    optimizer.step()

                    accuracy = Accuracy(output.data.numpy(), y.numpy())
                    trainLosses.append((loss.item(), accuracy))

                # Validation
                with torch.no_grad():
                    model.eval()
                    for x, y in validDataloader:
                        output = model(x.cuda()).cpu()
                        loss = criterion(output.double(), y.double())
                        accuracy = Accuracy(output.data.numpy(), y.numpy())

                        validLosses.append((loss.item(), accuracy))

                elapsedTime = et.gettime()

            trainLoss, trainAcc = map(np.mean, zip(*trainLosses))
            validLoss, validAcc = map(np.mean, zip(*validLosses))

            printEpoch(epoch, numEpochs, elapsedTime, trainLoss, trainAcc, validLoss, validAcc)
            history.append(((trainLoss, trainAcc), (validLoss, validAcc)))

            scheduler.step(validLoss)

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(modelDir, f'checkpoint_{epoch:03d}.wei'))

    torch.save(model.state_dict(), os.path.join(modelDir, 'final.wei'))
    pickleSave(history, os.path.join(modelDir, 'history.pkl'))

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
    parser.add_argument('-e', '--epochs', type = int,
            dest = 'epochs', default = 100,
            help = 'number of epochs (default: 100)')
    parser.add_argument('-l', '--learning_rate', type = float,
            dest = 'lr', default = 1e-3,
            help = 'learning rate (default: 1e-3)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with EventTimer('MAIN'):
        main()

