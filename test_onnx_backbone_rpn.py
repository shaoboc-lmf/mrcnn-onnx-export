import torch
import torchvision
from torchvision.ops._register_onnx_ops import _onnx_opset_version
import onnxruntime
import onnx

import io
import unittest

from . import models_original
from . import models_onnx_fixed

from .utils import *

assert _onnx_opset_version == 11

class BackboneRPNProxy(torch.nn.Module):

    def __init__(self,
                 backbone,
                 rpn):

        super(BackboneRPNProxy, self).__init__()
        self.backbone = backbone
        self.rpn      = rpn


    def forward(self,
                images_tensors,
                images_image_sizes,
                targets = None):
        # images_tensors:     tensor
        # images_image_sizes: list[tensor([h, w])]

        images   = ImageList(images_tensors,
                             images_image_sizes)

        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        return proposals

class TestONNX_Backbone_RPN(unittest.TestCase):

    def setUp(self):
        self.im = []
        for b in range(5):
            im_b= torch.load("dumped_data/im_b{}".format(b))
            self.im += [im_b]


        self.model_original   = models_original.detection.__dict__["maskrcnn_resnet50_fpn"](num_classes=91,
                                                                                            pretrained=True)
        self.model_onnx_fixed = models_onnx_fixed.detection.__dict__["maskrcnn_resnet50_fpn"](num_classes=91,
                                                                                              pretrained=True)

        self.model_original.eval()
        self.model_onnx_fixed.eval()

        self.backbone_rpn_onnx_mod = BackboneRPNProxy(self.model_onnx_fixed.backbone,
                                                 self.model_onnx_fixed.rpn)
        self.backbone_rpn_onnx_mod.eval()
        self.backbone_rpn_orig_mod = BackboneRPNProxy(self.model_original.backbone,
                                                 self.model_original.rpn)
        self.backbone_rpn_orig_mod.eval()

    @unittest.skip("Already passed, skip")
    @torch.no_grad()
    def test_backbone_rpn_torch(self):
        print("Details:")
        print("Test if the fixed model is working correctly in torch env")
        
        for b, im_b in enumerate(self.im):

            #Input
            images_tensors     = im_b["transformed_images.tensors"]
            images_image_sizes = im_b["transformed_images.image_sizes"]
            h, w               = images_image_sizes[0]
            images_image_sizes = [torch.tensor([h, w])]

            #Output
            proposals_g        = im_b["proposals"]
            
            proposals_o = self.backbone_rpn_onnx_mod(images_tensors,
                                                     images_image_sizes)


            for g, o in zip(proposals_g, proposals_o):
                self.assertEqual(g.shape, o.shape)
                print("Tensor size check ...ok!", g.shape, o.shape)
                try:
                    torch.testing.assert_allclose(g, o , rtol=1e-03, atol=1e-04)
                except AssertionError as e:
                    print("Tensor allclose failed", g.shape)
                    print(str(e))
                else:
                    print("Tensor close check ...ok!")

            print("...Batch {} ok\n".format(b))

    @torch.no_grad()
    def test_backbone_rpn_onnx(self):
        print("Details:")
        print("Test if the fixed model's onnx is working correctly in onnxruntime")

        images_tensors     = self.im[0]["transformed_images.tensors"]
        images_image_sizes = self.im[0]["transformed_images.image_sizes"]
        h, w               = images_image_sizes[0]
        images_image_sizes = [torch.tensor([h, w])]

        input_names  = ["images_tensors", "images_image_sizes"]
        output_names = ["proposals"]
        dynamic_axes = {"images_tensors": {2 : 'width', 3 : 'height'} }


        onnx_io = io.BytesIO()
        torch.onnx.export(self.backbone_rpn_onnx_mod,
                          (images_tensors, images_image_sizes,),
                          onnx_io,
                          do_constant_folding = True,
                          opset_version = 11,
                          export_params = True,
                          verbose = True,
                          input_names = input_names,
                          output_names = output_names,
                          dynamic_axes = dynamic_axes)

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        onnx_print_io_meta(ort_session)

        for b, im_b in enumerate(self.im):
            images_tensors     = im_b["transformed_images.tensors"]
            images_image_sizes = im_b["transformed_images.image_sizes"]
            h, w               = images_image_sizes[0]
            images_image_sizes = [torch.tensor([h, w])]
            
            #Input
            images_tensors_g     = to_numpy(images_tensors)
            images_image_sizes_g = to_numpy(images_image_sizes[0])

            #Output
            proposals_g        = im_b["proposals"]



            proposals_ort = ort_session.run(None,
                                            {
                                                "images_tensors"     : images_tensors_g,
                                                "images_image_sizes" : images_image_sizes_g
                                            })

            for g, ort in zip(proposals_g, proposals_ort):
                self.assertEqual(g.shape, ort.shape)
                print("Tensor size check ...ok!", g.shape, ort.shape)
                try:
                    torch.testing.assert_allclose(g, ort , rtol=1e-03, atol=1e-04)
                except AssertionError as e:
                    print("Tensor allclose failed", g.shape)
                    print(str(e))
                else:
                    print("Tensor close check ...ok!")

            print("...Batch {} ok\n".format(b))
