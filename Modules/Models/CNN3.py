from torch.nn import *
from Modules.Models.template import CNN_Template

class Model(CNN_Template):
    def __init__(self):
        def Conv(in_channel, out_channel, kernel_size, stride = 1):
            return Sequential(
                Conv2d(in_channel, out_channel, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2),
                BatchNorm2d(out_channel),
                ReLU(),
            )

        super().__init__()

        self.convolutional = Sequential(
            # Shape: (3, 64, 64)
            Conv(3, 64, 3),
            Conv(64, 96, 3),
            Conv(96, 128, 3),

            MaxPool2d(2, 2),

            # Shape: (128, 32, 32)

            Conv(128, 192, 3),
            Conv(192, 256, 3),
            MaxPool2d(2, 2),
            Dropout(0.4),

            # Shape: (256, 16, 16)

            Conv(256, 384, 3),
            Conv(384, 512, 3),
            MaxPool2d(2, 2),
            Dropout(0.4),

            # Shape: (512, 8, 8)

            Conv(512, 512, 3),
            MaxPool2d(2, 2),
            Dropout(0.5),

            # Shape: (512, 4, 4)

            AvgPool2d(4, 4),

            # Shape: (512, 1, 1)
        )

        self.fullyConnected = Sequential(
            # Shape: (512, 1, 1)
            Linear(512, 512),
            ReLU(),
            Dropout(0.5),


            Linear(512, 512),
            ReLU(),

            Linear(512, 1),
        )

        self.network = Sequential(
            self.convolutional,
            Flatten(),
            self.fullyConnected,
            Sigmoid()
        )



