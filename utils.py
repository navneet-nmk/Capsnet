import torch.nn.functional as F
import torch



def softmax(input, dim=1):
    """
    nn.functional.softmax does not take a dimension as of PyTorch version 0.2.0.
    This was created to add dimension support to the existing softmax function
    for now until PyTorch 0.4.0 stable is release.
    GitHub issue tracking this: https://github.com/pytorch/pytorch/issues/1020
    Arguments:
        input (Variable): input
        dim (int): A dimension along which softmax will be computed.
    """
    input_size = input.size()

    trans_input = input.transpose(dim, len(input_size) - 1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)

    return soft_max_nd.transpose(dim, len(input_size) - 1)



def one_hot_encode(target, length):
    """Converts batches of class indices to classes of one-hot vectors."""
    batch_s = target.size(0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec