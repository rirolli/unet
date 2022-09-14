import string
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    """
        This class return an encoder block of the UNet.
        A block is composed by:
        - 2 2D convolutional layer 3x3;
        - 2 non-linearity layer;
        - 1 max pooling layer.

        Parameters:
        - input_channels: input channels
        - output_channels: number of classes
        - b_dropout: add the dropout layer (default = True)
        - p: probability of dropout
    """

    def __init__(self, input_channels:int, output_channels:int, b_dropout=True, p=0.2):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.b_dropout = b_dropout

        # layer of the block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2)
        if self.b_dropout:
            self.drop = nn.Dropout2d


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp(x)
        if self.b_dropout:
            x = self.drop(x)
        
        return x

class DecoderBlock(nn.Module):

    def __init__(self, input_channels:int, output_channels:int, b_dropout=True, p=0.2, mode="upsample"):
        """
        mode: can be "upsample" or "convtranspose"
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.b_dropout = b_dropout
        self.mode = mode

        if mode=="upsample":
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode=="convtranspose":
            nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels*2, kernel_size=2, bias=False)


    def forward(self, x):
        pass

class BlockFactory():
    """
        This class is needed to create the blocks for the UNet. Allows decoupling between objects.
    """

    def __init__(self) -> None:
        pass

    def getEncoderBlock(self, input_channels:int, output_channels:int, b_dropout=True, p=0.2) -> nn.Module:
        """Return an Encoder block."""
        return EncoderBlock(input_channels, output_channels, b_dropout, p)

    def getDecoderBlock(self, input_channels:int, output_channels:int, b_dropout=True, p=0.2) -> nn.Module:
        """Return a Decoder block."""
        return DecoderBlock(input_channels, output_channels, b_dropout, p)

    def get(self, block:string, input_channels:int, output_channels:int, b_dropout=True, p=0.2) -> nn.Module:
        """
        Return a block for the UNet.
        Parameter:
        - block: is the type of block that you want to return. There are two types: "encoder" and "decoder" blocks.
        - input_channels: input channels
        - output_channels: number of classes
        - b_dropout: add the dropout layer (default = True)
        - p: probability of dropout
        """
        if block=="encoder":
            return self.getEncoderBlock(input_channels, output_channels, b_dropout, p)
        elif block=="decoder":
            return self.getDecoderBlock(input_channels, output_channels, b_dropout, p)
        elif block is None:
            raise Exception("Block type must be specified.")
        else:
            raise Exception(f"Block type \"{block}\" is not valid.")

class UNet:
    # TO-DO
    pass