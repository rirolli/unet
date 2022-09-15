import string
import torch
import torch.nn as nn
import torchvision.transforms as T

class DoubleConvolutionalLayer(nn.Module):
    """This class create a double convolutional layer with the following sequential structure:
    - Convolutional Layer 1;
    - ReLu Layer 1;
    - Convolutional Layer 2;
    - ReLu Layer 2;
    """

    def __init__(self, input_channels:int, output_channels:int, middle_channels:int=None, kernel_size=3, padding=1, bias=False) -> None:
        super().__init__()
        if middle_channels is None:
            middle_channels = output_channels

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=middle_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=middle_channels, out_channels=output_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x



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
        - middle_channels: number of classes for intermediate passages
        - b_dropout: add the dropout layer (default = True)
        - p: probability of dropout
    """

    def __init__(self, input_channels:int, output_channels:int=None, middle_channels:int=None, b_dropout=True, p=0.2, is_last_block=False) -> None:
        super().__init__()

        # defaults
        if middle_channels is None:
            middle_channels = output_channels
        if b_dropout is None:
            b_dropout = True
        if p is None:
            p=0.2
        if is_last_block is None:
            is_last_block = False
        if output_channels is None:
            output_channels = input_channels*2  # the number of output channels is duplicated

        # initialize the variables
        self.b_dropout = b_dropout
        self.is_last_block = is_last_block

        # layer of the block
        self.d_conv = DoubleConvolutionalLayer(input_channels=input_channels, middle_channels=middle_channels, output_channels=output_channels)
        if not self.is_last_block:
            self.mp = nn.MaxPool2d(kernel_size=2)
        if self.b_dropout:
            self.drop = nn.Dropout2d(p=p)


    def forward(self, x):
        x = self.d_conv(x)
        if not self.is_last_block:
            x = self.mp(x)
        if self.b_dropout:
            x = self.drop(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, input_channels:int, output_channels:int=None, middle_channels:int=None, b_dropout=True, p=0.2, mode="upsample"):
        """
        This class r4etur a Decoder Block for the UNet.

        - mode: can be "upsample" or "convtranspose"
        """
        super().__init__()

        # defaults
        if output_channels is None:
            output_channels = input_channels // 2
        if middle_channels is None:
            middle_channels = output_channels
        if b_dropout is None:
            b_dropout = True
        if p is None:
            p=0.2
        if mode is None:
            mode="upsample"

        # initialize the variables
        self.b_dropout = b_dropout
        self.mode = mode

        if middle_channels is None:
            middle_channels = output_channels

        halved_channels = input_channels // 2  # In the decoder process the number of channels must be halved after the upsampling

        if mode=="upsample":
            self.upscale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels=input_channels, out_channels=halved_channels, kernel_size=2, bias=False)
            )
        elif mode=="convtranspose":
            self.upscale = nn.ConvTranspose2d(in_channels=input_channels, out_channels=halved_channels, kernel_size=2, bias=False)

        # the input_channels number have to be the same of the start because in this second step the halved output is concatenated
        # with the corresponding encoder output layer.
        self.d_conv = DoubleConvolutionalLayer(input_channels=input_channels, middle_channels=middle_channels, output_channels=output_channels)

        if self.b_dropout:
            self.drop = nn.Dropout2d(p=p)

    def forward(self, x_encoder, x_decoder):
        # first we have to upsampling the input and halves the channels
        x_decoder_halved = self.upscale(x_decoder)

        # crops the encoder output
        h_decoder = x_decoder_halved.size()[2]
        w_decoder = x_decoder_halved.size()[3]
        x_encoder_cropped = T.CenterCrop(size=(h_decoder, w_decoder))(x_encoder)

        # concatenate the halved decoder input with the cropped encoder output
        x = torch.cat([x_encoder_cropped, x_decoder_halved], dim=1)

        # go on with the double convolutional
        x = self.d_conv(x)

        # dropout layer
        if self.b_dropout:
            x = self.drop(x)

        return x


class BlockFactory():
    """
        This class is needed to create the blocks for the UNet. Allows decoupling between objects.
    """

    def __init__(self) -> None:
        pass

    def getEncoderBlock(self, input_channels:int, output_channels:int=None, middle_channels:int=None, b_dropout=True, p=0.2, is_last_block=False) -> nn.Module:
        """Return an Encoder block."""
        return EncoderBlock(input_channels, output_channels, middle_channels, b_dropout, p, is_last_block)

    def getDecoderBlock(self, input_channels:int, output_channels:int=None, middle_channels:int=None, b_dropout=True, p=0.2, is_last_block=False) -> nn.Module:
        """Return a Decoder block."""
        return DecoderBlock(input_channels, output_channels, middle_channels, b_dropout, p, is_last_block)

    def getBlock(self, block:string, input_channels:int, output_channels:int=None, b_dropout=True, p=0.2, is_last_block=False) -> nn.Module:
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
            return self.getEncoderBlock(input_channels, output_channels, b_dropout, p, is_last_block)
        elif block=="decoder":
            return self.getDecoderBlock(input_channels, output_channels, b_dropout, p)
        elif block is None:
            raise Exception("Block type must be specified.")
        else:
            raise Exception(f"Block type \"{block}\" is not valid.")

class UNet (nn.Module):
    
    def __init__(self, n_channels:int, n_classes:int) -> None:
        super().__init__()
        # The number of the variables is the depth in the architecture.

        # initialization
        block_factory = BlockFactory()

        # Encoder
        self.b_enc1 = block_factory.getEncoderBlock(input_channels=n_channels, output_channels=64) # output = 64 classes
        self.b_enc2 = block_factory.getEncoderBlock(input_channels=64) # output = 128 classes
        self.b_enc3 = block_factory.getEncoderBlock(input_channels=128) # output = 256 classes
        self.b_enc4 = block_factory.getEncoderBlock(input_channels=256) # output = 512 classes

        # Intermediate
        self.b_union = block_factory.getEncoderBlock(input_channels=512, is_last_block=True) # output = 1024 classes

        # Decoder
        self.b_dec4 = block_factory.getDecoderBlock(input_channels=1024) # output = 512 classes
        self.b_dec3 = block_factory.getDecoderBlock(input_channels=512) # output = 256 classes
        self.b_dec2 = block_factory.getDecoderBlock(input_channels=256) # output = 128 classes
        self.b_dec1 = block_factory.getDecoderBlock(input_channels=128) # output = 64 classes

        # 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.b_enc1(x)
        x2 = self.b_enc2(x1)
        x3 = self.b_dec3(x2)
        x4 = self.b_enc4(x3)
        x = self.b_union(x4)
        x = self.b_dec4(x4,x)
        x = self.b_enc3(x3,x)
        x = self.b_enc2(x2,x)
        x = self.b_enc1(x1,x)
        x = self.conv1x1(x)
        return x