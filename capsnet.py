import torch
import torch.functional as F
from torch import nn
from torch.autograd import Variable
from capsule import Capsule



class Capsnet(nn.Module):

    # A simple capsnet with 3 layers
    def __init__(self, num_input_conv_layer, num_output_conv_layer, conv_kernel_dim, conv_kernel_stride, num_primary_unit, primary_unit_size,
                 num_classes, output_unit_size, num_routing, cuda_enabled, regularization_scale):

        super(Capsnet, self).__init__()

        self.cuda_enabled = cuda_enabled

        # MNIST Parameters
        self.image_width = 28
        self.image_height = 28
        self.image_channels = 1

        self.regularization = regularization_scale # SSE regularization scale

        # Layer 1 : Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=num_input_conv_layer, out_channels=num_output_conv_layer,
                               padding=0, kernel_size=conv_kernel_dim, stride=conv_kernel_stride)
        self.relu = nn.ReLU(inplace=True)


        # Primary Layer
        # Conv2d with squash activation
        self.primary = Capsule(in_unit=0,
                               in_channel=num_output_conv_layer,
                               num_unit=num_primary_unit,
                               unit_size=primary_unit_size,
                               use_routing=False,
                               num_routing=num_routing,
                               cuda_enabled=cuda_enabled,
                               conv_kernel_size=9,
                               num_conv_output=32,
                               conv_kernel_stride=2
                               )

        # DigitCaps layer
        # Capsule layer with dynamic routing
        self.digits = Capsule(in_unit=num_primary_unit,
                              in_channel=primary_unit_size,
                              num_unit=num_classes,
                              unit_size=output_unit_size,  # 16D capsule per digit class
                              use_routing=True,
                              num_routing=num_routing,
                              cuda_enabled=cuda_enabled,
                              conv_kernel_size=9,
                              num_conv_output=32,
                              conv_kernel_stride=2
                              )



    def forward(self, x):
        # Forward Pass
        conv1 = self.conv1(x)
        conv1 = self.relu(conv1)

        primary_caps_out = self.primary(conv1)
        output = self.digits(primary_caps_out)

        return output







