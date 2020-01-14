import torch
import torchvision
from torchvision.ops._register_onnx_ops import _onnx_opset_version
import onnxruntime
import onnx

import io
import unittest
import types

from .utils import *

from . import models_original
from .patches.rpn import concat_box_prediction_layers, anchor_generator_forward_patch

assert _onnx_opset_version == 11

class TestONNXRpn(unittest.TestCase):

    def setUp(self):
        self.im = []
        for b in range(5):
            im_b= torch.load("dumped_data/im_b{}".format(b))
            self.im += [im_b]
        
        self.model = models_original.detection.__dict__["maskrcnn_resnet50_fpn"](num_classes=91,
                                                                                 pretrained=True)
        self.model.eval()

    @torch.no_grad()
    def test_rpn_head_concat(self):
        '''
        features-> {0 : tensor(1, 256, h, w), 
                    1 : tensor(1, 256, h, w), 
                    2 : tensor(1, 256, h, w), 
                    3 :, 
                    pool}
          (head)
        objectness-> [tensor(1, 3, h, w)] x 5
        pred_bbox_deltas_g-> [tensor(1, 4, h, w)] x 5
          (concat)
        objectness-> tensor(obj, 1)
        pred_bbox_deltas_g-> tensor(obj, 4)
        '''

        rpn_head = self.model.rpn.head

        class RpnHeadProxy(torch.nn.Module):

            def __init__(self,
                         head):

                super(RpnHeadProxy, self).__init__()
                self.head = head

            def forward(self, features):

                features = list(features.values())
                objectness, pred_bbox_deltas = self.head(features)
                return concat_box_prediction_layers(objectness, pred_bbox_deltas)

        rpn_head_mod = RpnHeadProxy(rpn_head)
        rpn_head_mod.eval()
        
        features_g = self.im[0]["features"]
        onnx_io = io.BytesIO()
        input_names    = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_pool"]
        output_names   = ["objectness", "pred_bbox_deltas"]
        dynamic_axes   = { name : {2 : 'width', 3 : 'height'}   for name in input_names }
        dynamic_axes.update({name : {0 : 'objs'} for name in output_names })
        torch.onnx.export(rpn_head_mod,
                          (features_g,),
                          onnx_io,
                          do_constant_folding = True,
                          opset_version = 11,
                          export_params = True,
                          input_names  = input_names,
                          output_names = output_names,
                          dynamic_axes = dynamic_axes,
                          verbose = True)
        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        
        # onnx_print_io_meta(ort_session)
        
        for b, im_b in enumerate(self.im):
            features_g                       = im_b["features"]
            objectness_g, pred_bbox_deltas_g = rpn_head_mod(features_g)

            features_g = {k : to_numpy(v) for k, v in features_g.items()}            
            res_g = [to_numpy(objectness_g), to_numpy(pred_bbox_deltas_g)]

            res_ort = ort_session.run(None,
                                      {
                                          "feat_0" : features_g["0"],
                                          "feat_1" : features_g["1"],
                                          "feat_2" : features_g["2"],
                                          "feat_3" : features_g["3"],
                                          "feat_pool" : features_g["pool"]
                                      })

            for g, ort in zip(res_g, res_ort):
                self.assertEqual(g.shape, ort.shape)
                torch.testing.assert_allclose(g, ort, rtol=1e-03, atol=1e-05)

            print("...Batch {} ok".format(b))

    @torch.no_grad()
    def test_rpn_anchor_generator(self):
        anchor_generator = self.model.rpn.anchor_generator

        class AnchorGeneratorProxy(torch.nn.Module):
            def __init__(self,
                         a_g,
                         forward_patch):

                super(AnchorGeneratorProxy, self).__init__()
                self.anchor_generator = a_g
                self.anchor_generator.forward = types.MethodType(forward_patch,
                                                                 self.anchor_generator)

            def forward(self,
                        images_tensors,
                        images_image_sizes,
                        features):
                features = list(features.values())
                anchors  = self.anchor_generator(images_tensors,
                                                 images_image_sizes,
                                                 features)
                return anchors

        anchor_generator_mod = AnchorGeneratorProxy(anchor_generator,
                                                    anchor_generator_forward_patch)
        anchor_generator_mod.eval()

        features_g = self.im[0]["features"]
        transformed_images_tensors_g     = self.im[0]["transformed_images.tensors"]
        transformed_images_image_sizes_g = self.im[0]["transformed_images.image_sizes"] # list of tuple
        h, w = transformed_images_image_sizes_g[0]
        transformed_images_image_sizes_g = [torch.tensor([h, w])]
        
        onnx_io = io.BytesIO()
        # for anchors over all feature maps
        # input_names  = ["image_list_tensors", "feat_0", "feat_1", "feat_2", "feat_3", "feat_pool"]
        # output_names = ["anchor_0", "anchor_1", "anchor_2", "anchor_3", "anchor_pool"]
        # dynamic_axes = { name : {2 : 'width', 3 : 'height'} for name in input_names }
        # dynamic_axes.update({name : {0 : 'anchor'} for name in output_names})

        # for anchors
        input_names  = ["image_list_tensors", "image_list_image_sizes",
                        "feat_0", "feat_1", "feat_2", "feat_3", "feat_pool"]
        output_names = ["anchor"]
        dynamic_axes = { name : {2 : 'width', 3 : 'height'} for name in input_names }
        dynamic_axes.update({name : {0 : 'anchor'} for name in output_names})
        torch.onnx.export(anchor_generator_mod,
                          (transformed_images_tensors_g,
                           transformed_images_image_sizes_g, #loop is unrolled, don't need
                           features_g, ),
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

            features_g                       = im_b["features"]
            transformed_images_tensors_g     = im_b["transformed_images.tensors"]
            transformed_images_image_sizes_g = self.im[0]["transformed_images.image_sizes"] # list of tuple
            h, w = transformed_images_image_sizes_g[0]
            transformed_images_image_sizes_g = [torch.tensor([h, w])]
            anchors_g = anchor_generator_mod(transformed_images_tensors_g,
                                             transformed_images_image_sizes_g,
                                             features_g)

            features_g = {k : to_numpy(v) for k, v in features_g.items()}
            transformed_images_tensors_g  = to_numpy(transformed_images_tensors_g)
            anchors_g  = [to_numpy(a) for a in anchors_g]

            anchors_ort = ort_session.run(None,
                                          {
                                              "image_list_tensors" : transformed_images_tensors_g,
                                              "feat_0"      : features_g["0"],
                                              "feat_1"      : features_g["1"],
                                              "feat_2"      : features_g["2"],
                                              "feat_3"      : features_g["3"],
                                              "feat_pool"   : features_g["pool"]
                                          })

            for g, ort in zip(anchors_g, anchors_ort):
                self.assertEqual(g.shape, ort.shape)
                torch.testing.assert_allclose(g, ort, rtol=1e-03, atol=1e-05)
            
            print("...Batch {} ok".format(b))


    @torch.no_grad()
    def test_rpn_head_anchor_generator(self):

        rpn_head         = self.model.rpn.head
        anchor_generator = self.model.rpn.anchor_generator
        box_coder        = self.model.rpn.box_coder

        class HeadAnchorGenProxy(torch.nn.Module):

            def __init__(self,
                         head,
                         anchor_gen,
                         anchor_gen_forward_patch,
                         box_coder):

                super(HeadAnchorGenProxy, self).__init__()
                self.head                     = head
                self.anchor_generator         = anchor_gen
                self.anchor_generator.forward = types.MethodType(anchor_gen_forward_patch,
                                                                 self.anchor_generator)
                self.box_coder                = box_coder

            def forward(self,
                        images_tensors,
                        images_image_sizes,
                        features):

                features = list(features.values())
                
                objectness, pred_bbox_deltas = self.head(features)
                num_anchors_per_level = [o[0].numel() for o in objectness] # of anchors / feat level before concat
                objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

                anchors  = self.anchor_generator(images_tensors,
                                                 images_image_sizes,
                                                 features)
                
                proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
                num_images = len(images_image_sizes)
                proposals = proposals.view(num_images, -1, 4) #N can not be dynamic dimension for now

                return objectness, proposals

        head_anchor_gen_mod = HeadAnchorGenProxy(rpn_head,
                                                 anchor_generator,
                                                 anchor_generator_forward_patch,
                                                 box_coder)

        features_g = self.im[0]["features"]
        transformed_images_tensors_g     = self.im[0]["transformed_images.tensors"]
        transformed_images_image_sizes_g = self.im[0]["transformed_images.image_sizes"] # list of tuple
        h, w = transformed_images_image_sizes_g[0]
        transformed_images_image_sizes_g = [torch.tensor([h, w])]
        
        input_names  = ["image_list_tensors", "image_list_image_sizes",
                        "feat_0", "feat_1", "feat_2", "feat_3", "feat_pool"]
        
        output_names = ["objectness", "proposals"]
        dynamic_axes = { name : {2 : 'width', 3 : 'height'} for name in ["image_list_tensors",
                                                                         "feat_0",
                                                                         "feat_1",
                                                                         "feat_2",
                                                                         "feat_3",
                                                                         "feat_pool"]}
        dynamic_axes.update({"objectness" : {0 : 'anchors_all_image'} , "proposals" : {1 : 'anchors'}})
        # output is a list of tenosr with shep 1000, 4
        
        onnx_io = io.BytesIO()
        torch.onnx.export(head_anchor_gen_mod,
                          (transformed_images_tensors_g,
                           transformed_images_image_sizes_g, #loop is unrolled, don't need
                           features_g, ),
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

            features_g                       = im_b["features"]
            transformed_images_tensors_g     = im_b["transformed_images.tensors"]
            transformed_images_image_sizes_g = im_b["transformed_images.image_sizes"] # list of tuple
            h, w = transformed_images_image_sizes_g[0]
            transformed_images_image_sizes_g = [torch.tensor([h, w])]
            
            objs_g, proposals_g = head_anchor_gen_mod(transformed_images_tensors_g,
                                                      transformed_images_image_sizes_g,
                                                      features_g)
            
            transformed_images_tensors_g  = to_numpy(transformed_images_tensors_g)
            features_g                    = {k : to_numpy(v) for k, v in features_g.items()}


            objs_g      = to_numpy(objs_g)
            proposals_g = to_numpy(proposals_g)

            res_g = [objs_g, proposals_g]
            
            objs_ort, proposals_ort \
                = ort_session.run(None,
                                  {
                                      "image_list_tensors" : transformed_images_tensors_g,
                                      "feat_0"      : features_g["0"],
                                      "feat_1"      : features_g["1"],
                                      "feat_2"      : features_g["2"],
                                      "feat_3"      : features_g["3"],
                                      "feat_pool"   : features_g["pool"]
                                  })
            res_ort = [objs_ort, proposals_ort]

            for g, ort in zip(res_g, res_ort):
                print(ort.shape)
                self.assertEqual(g.shape, ort.shape)
                torch.testing.assert_allclose(g, ort , rtol=1e-03, atol=1e-04) # released from -05 to -04

            print("...Batch {} ok".format(b))
