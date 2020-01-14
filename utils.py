
from __future__ import division

import torch
from torch.jit.annotations import List, Tuple
from torch import Tensor

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]])
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()    

def onnx_print_io_meta(ort_session):

    print("\n==ort inputs==")
    for ort_inputs_meta in ort_session.get_inputs():
        # print(dir(ort_inputs_meta))
        print(ort_inputs_meta.name)
        print(ort_inputs_meta.shape)
        print(ort_inputs_meta.type)
        print()

    print("\n==ort outputs==")
    for ort_outputs_meta in ort_session.get_outputs():
        # print(dir(ort_outputs_meta))
        print(ort_outputs_meta.name)
        print(ort_outputs_meta.shape)
        print(ort_outputs_meta.type)
        print()
