import os, cv2
import numpy as np

from sklearn.model_selection import train_test_split

SEED = 0x06902029

# Training Data
cats, dogs = [], []
for i in range(12500):
    cats.append(cv2.resize(cv2.imread(os.path.join('DATA/train', f'cat.{i}.jpg')), (64, 64)))
    dogs.append(cv2.resize(cv2.imread(os.path.join('DATA/train', f'dog.{i}.jpg')), (64, 64)))

X = np.stack(cats + dogs)
Y = np.stack([0] * 12500 + [1] * 12500).reshape(-1, 1).astype(np.float64)

trainX, validX, trainY, validY = train_test_split(X, Y, test_size = 0.2, random_state = SEED)

print(trainX.shape, trainY.shape)
np.save('DATA/trainX.npy', trainX)
np.save('DATA/trainY.npy', trainY)

print(validX.shape, validY.shape)
np.save('DATA/validX.npy', validX)
np.save('DATA/validY.npy', validY)

# Testing Data
imgs = []
for i in range(1, 12501):
    imgs.append(cv2.resize(cv2.imread(os.path.join('DATA/test/', f'{i}.jpg')), (64, 64)))

testX = np.stack(imgs)
print(testX.shape)

np.save('DATA/testX.npy', testX)

