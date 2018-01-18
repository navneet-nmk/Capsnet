import torch
import torch.functional as F
from torch import nn
from torch.autograd import Variable


class Capsule(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, input_units, num_input_channels, num_output_channels, num_iterations, use_routing, cuda_enabled,
                 conv_kernel_size=None, conv_strides=None):

        super(Capsule, self).__init__()

        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_iterations = num_iterations
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.use_routing = use_routing
        self.cuda_enabled = cuda_enabled
        self.input_units = input_units

        if self.use_routing:
            """
                        Based on the paper, DigitCaps which is capsule layer(s) with
                        capsule inputs use a routing algorithm that uses this weight matrix, Wij
            """
            self.route_weights = nn.Parameter(torch.randn(1, num_input_channels, num_output_channels, num_capsules, input_units))
        else:
            self.capsules =  nn.ModuleList([nn.Conv2d(num_input_channels, num_output_channels, padding=0, stride=conv_strides)
                                            for _ in range(num_output_channels)])

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

        for iteration in range(self.num_iterations):
            


