import torch
import torchvision
from torchvision.ops._register_onnx_ops import _onnx_opset_version
import onnxruntime
import onnx

import io
import unittest

from .utils import *

from . import models_original

assert _onnx_opset_version == 11


class TestONNXBackbone(unittest.TestCase):


    def setUp(self):
        self.im = []
        for b in range(5):
            im_b= torch.load("dumped_data/im_b{}".format(b))
            self.im += [im_b]
        
        self.model = models_original.detection.__dict__["maskrcnn_resnet50_fpn"](num_classes=91,
                                                                                 pretrained=True)
        self.model.eval()

    @torch.no_grad()
    def test_backbone(self):

        # input
        transformed_images_tensors_g = self.im[0]["transformed_images.tensors"]

        # module
        backbone = self.model.backbone

        onnx_io = io.BytesIO()
        torch.onnx.export(backbone,
                          (transformed_images_tensors_g),
                          onnx_io,
                          do_constant_folding = True,
                          opset_version = 11,
                          export_params = True,
                          input_names = ["transformed_images_tensors"],
                          output_names = ["features_0",
                                          "features_1",
                                          "features_2",
                                          "features_3",
                                          "features_pool"],
                          dynamic_axes = { "transformed_images_tensors" : { 2 : 'width',
                                                                            3 : 'height'},
                                           "features_0"    : { 2 : 'width',
                                                               3 : 'height'},
                                           "features_1"    : { 2 : 'width',
                                                               3 : 'height'},
                                           "features_2"    : { 2 : 'width',
                                                               3 : 'height'},
                                           "features_3"    : { 2 : 'width',
                                                               3 : 'height'},
                                           "features_pool" : { 2 : 'width',
                                                               3 : 'height'},
                          },
                          verbose = True)

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())        

        onnx_print_io_meta(ort_session)

        for b, im_b in enumerate(self.im):
            print("...Checking batch {}".format(b))
            transformed_images_tensors_g = im_b["transformed_images.tensors"]
            features_g                   = im_b["features"]

            transformed_images_tensors_g = to_numpy(transformed_images_tensors_g)
            features_g                   = [to_numpy(v) for v in features_g.values()]            

            features_ort = ort_session.run(None, {"transformed_images_tensors" : transformed_images_tensors_g})

            for ort, golden in zip(features_ort, features_g):
                self.assertEqual(ort.shape, golden.shape)
                torch.testing.assert_allclose(ort, golden, rtol=1e-03, atol=1e-05)

            print("...batch {} ok".format(b))
