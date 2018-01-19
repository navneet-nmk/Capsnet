import torch
import torch.functional as F
from torch import nn
from torch.autograd import Variable


class Capsule(nn.Module):
    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing, num_routing, cuda_enabled, conv_kernel_size,num_conv_output, conv_kernel_stride):

        super(Capsule, self).__init__()

        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.use_routing = use_routing
        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled
        self.conv_kernel_size = conv_kernel_size
        self.conv_kernel_stride = conv_kernel_stride
        self.num_conv_output = num_conv_output


        if self.use_routing:
            """
            Based on the paper, DigitCaps which is capsule layer(s) with
            capsule inputs use a routing algorithm that uses this weight matrix, Wij
            """
            # weight shape:
            # [1 x primary_unit_size x num_classes x output_unit_size x num_primary_unit]
            # == [1 x 1152 x 10 x 16 x 8]
            self.weight = nn.Parameter(torch.randn(1, in_channel, num_unit, unit_size, in_unit))
        else:
            """
            According to the CapsNet architecture section in the paper,
            we have routing only between two consecutive capsule layers (e.g. PrimaryCapsules and DigitCaps).
            No routing is used between Conv1 and PrimaryCapsules.
            This means PrimaryCapsules is composed of several convolutional units.
            """
            # Define 8 convolutional units.
            self.conv_units = nn.ModuleList([
                nn.Conv2d(self.in_channel, self.num_conv_output, self.conv_kernel_size, self.conv_kernel_stride) for u in range(self.num_unit)
            ])

    @staticmethod
    def squash(tensor, dim=-1):

        # Squashing method as described in the paper
        # Non linear squashing function so that short vectors are squashed to 0 and long vectors get shrunk to a val lower than 1\
        norm =  (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = norm / (1+norm)
        return scale * tensor/torch.sqrt(norm)

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)


    def routing(self, x):

        """
                Routing algorithm for capsule.
                :input: tensor x of shape [128, 8, 1152]
                :return: vector output of capsule j
        """


        batch_size = x.size(0)
        x = x.transpose(1,2)
        x = torch.stack([x] * self.num_output_channels, dim=2).unsqueeze(4)

        batch_weight = torch.cat([self.route_weights] * batch_size, dim=0)

        u_hat = torch.matmul(batch_weight, x)

        # All the routing logits (b_ij in the paper) are initialized to zero.
        b_ij = Variable(torch.zeros(1, self.num_input_channels, self.num_output_channels, 1))
        if self.cuda_enabled:
            b_ij = b_ij.cuda()


        # Paper uses 3 routing iterations

        for iteration in range(self.num_iterations):
            # Routing algorithm

            # Calculate routing coefficients
            c_ij = utils.softmax(b_ij, dim=2)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)


            # s_j is total input to a capsule, is a weigthed sum over all "prediction vectors".
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            v_j = self.squash(s_j, dim=3)
            # in_channel is 1152.
            # v_j1 shape: [128, 1152, 10, 16, 1]
            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)

            #The agreement - vj1 matmul u_hat
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update routing (b_ij) by adding the agreement to the initial logit.
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)


    def no_routing(self, x):
        # Create 8 convolutional unit.
        # A convolutional unit uses normal convolutional layer with a non-linearity (squash).
        unit = [self.conv_units[i](x) for i, l in enumerate(self.conv_units)]

        # Stack all unit outputs.
        # Stacked of 8 unit output shape: [128, 8, 32, 6, 6]
        unit = torch.stack(unit, dim=1)

        batch_size = x.size(0)

        # Flatten the 32 of 6x6 grid into 1152.
        # Shape: [128, 8, 1152]
        unit = unit.view(batch_size, self.num_unit, -1)

        # Add non-linearity
        # Return squashed outputs of shape: [128, 8, 1152]

        return self.squash(unit, dim=2)  # dim 2 is the third dim (1152D array) in our tensor
            


