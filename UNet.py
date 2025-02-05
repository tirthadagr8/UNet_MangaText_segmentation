import torch
import torch.nn as nn
from model import Model
from config import Config
import torch.nn.functional as F
from utility import *


class UNet(Model):
    """
    UNet architecture for image segmentation with reduced parameters.
    Inputs:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels=3, out_channels=1, config=None):
        super().__init__()
        
        self.img_size=config.img_size if config is not None else 1024
        # Encoder (Downsampling path)
        self.encoder1 = self._block(in_channels, 32, "enc1")  # Reduced channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(32, 64, "enc2")  # Reduced channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(64, 128, "enc3")  # Reduced channels
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(128, 256, "enc4")  # Reduced channels
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck1 = self._block(256, 512, "bottleneck1")  # Reduced channels
        self.bottleneck2 = self._block(512, 512, "bottleneck2")
        # Decoder (Upsampling path)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._block(512, 256, "dec4")

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._block(256, 128, "dec3")

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._block(128, 64, "dec2")

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self._block(64, 32, "dec1")

        # Output layer
        self.output_layer = nn.Conv2d(32, out_channels, kernel_size=1)
        
        if config is not None:
            print("Loading Model Weights")
            self.load(config.model_checkpoint, config.device)

    def _block(self, in_channels, out_channels, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # Reduced dropout rate
        )

    def forward(self, pixel_values,masks=None,**kwargs):
        # Encoder
        enc1 = self.encoder1(pixel_values)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck1(self.pool4(enc4))
        bottleneck = self.bottleneck2(bottleneck)

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output
        output = self.output_layer(dec1)
        # output = torch.sigmoid(output)
        loss=None
        if masks is not None:
            loss=self.compute_loss(output,masks)
        return   {
            'loss':loss,
            'logits':output
        }


    def compute_loss(self,predicted_mask,input_mask):
        return combined_dice_bce_loss(predicted_mask,input_mask,self.img_size)


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = 64, 64
    model = UNet(in_channels=3, out_channels=1,img_size=64).to(device)

    # Test forward pass
    x = torch.randn(1, 3, img_h, img_w).to(device)  # Batch of 2 images
    z = torch.rand(1, 1, img_h, img_w).to(device)
    y = model(x,z)
    print("Output shape:", y['logits'].shape)  # Expected: (2, 1, 512, 512)
    print("Output shape:", y['loss'])
    print(f"Trainable parameters: {count_parameters(model)/1e6:.1f}M")