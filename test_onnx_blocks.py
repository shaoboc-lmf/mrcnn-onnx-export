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

class TestONNXModule(unittest.TestCase):

    def setUp(self):
        self.im = []
        for b in range(5):
            im_b= torch.load("dumped_data/im_b{}".format(b))
            self.im += [im_b]
        
        self.model = models_original.detection.__dict__["maskrcnn_resnet50_fpn"](num_classes=91,
                                                                                 pretrained=True)
        self.model.eval()


    @unittest.skip("transform fail, needs more debugging")
    @torch.no_grad()
    def test_transform(self):

        # input
        images_g                         = self.im["images"]
        print(type(images_g))
        
        # output
        transformed_images_tensors_g     = self.im["transformed_images.tensors"]
        transformed_images_image_sizes_g = self.im["transformed_images.image_sizes"]

        # module
        transform = self.model.transform

        # export onnx
        onnx_io = io.BytesIO()
        torch.onnx.export(transform,
                          (images_g,), #inputs
                          onnx_io,
                          do_constant_folding = True,
                          opset_version = 11,
                          export_params = True,
                          input_names   = ["images"],
                          verbose = True)

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())

        onnx_print_io_meta(ort_session)

        images_g, _ = torch.jit._flatten(images_g)
        
        images_g = list(map(to_numpy, images_g))
        
        ort_output = ort_session.run(None, {"images" : images_g})

        # print("fuck")
        # for o in ort_output:
        #     print(o.name)
        #     print(o.shape)
        #     print(o.type)

    @unittest.skip("original rpn can not export correctly, issues are been fixed and tested other test_rpn*.py")
    @torch.no_grad()
    def test_rpn(self):

        rpn = self.model.rpn

        for b, im_b in enumerate(self.im):
            print("...Checking batch {}".format(b))            
            # inputs
            features_g                       = im_b["features"]
            transformed_images_tensors_g     = im_b["transformed_images.tensors"]
            transformed_images_image_sizes_g = im_b["transformed_images.image_sizes"]
            transformed_images_image_sizes_g = torch.tensor(transformed_images_image_sizes_g)
            
            # outputs
            proposals_g                      = im_b["proposals"]

            # proposals, _ = rpn(transformed_images_tensors_g,
            #                    transformed_images_image_sizes_g,
            #                    features_g, None)

            proposals, _ = rpn(ImageList(transformed_images_tensors_g,
                                         transformed_images_image_sizes_g),
                               features_g, None)

            self.assertTrue(len(proposals) == 1 == len(proposals_g))
            torch.testing.assert_allclose(proposals[0],
                                          proposals_g[0],
                                          rtol=1e-03, atol=1e-05)
            
        # # inputs
        # features_g                       = self.im[0]["features"]
        # transformed_images_tensors_g     = self.im[0]["transformed_images.tensors"]
        # transformed_images_image_sizes_g = self.im[0]["transformed_images.image_sizes"]
        # transformed_images_image_sizes_g = torch.tensor(transformed_images_image_sizes_g)
        
        # onnx_io = io.BytesIO()
        # torch.onnx.export(rpn,
        #                   (transformed_images_tensors_g,
        #                    transformed_images_image_sizes_g,                           
        #                    features_g,),
        #                   onnx_io,
        #                   do_constant_folding = True,
        #                   opset_version = 11,
        #                   export_params = True,
        #                   input_names = ["transformed_images_tensors",
        #                                  "transformed_images_image_sizes",
        #                                  "features"],
        #                   output_names = ["proposals"],
        #                   verbose = True)

        # ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())        

        # onnx_print_io_meta(ort_session)        
        
        # for b, im_b in enumerate(self.im):
        #     print("...Checking batch {}".format(b))            
        #     # inputs
        #     features_g                       = im_b["features"]
        #     transformed_images_tensors_g     = im_b["transformed_images.tensors"]
        #     transformed_images_image_sizes_g = im_b["transformed_images.image_sizes"]
        #     transformed_images_image_sizes_g = torch.tensor(transformed_images_image_sizes_g)
            
        #     # outputs
        #     proposals_g                      = im_b["proposals"]

        #     proposals_ort = ort_session.run(None,
        #                                     {"transformed_images_tensors" : transformed_images_tensors_g,
        #                                      "transformed_images_image_sizes" : transformed_images_image_sizes_g,
        #                                      "features" : features_g})

        #     break
