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
from .patches.rpn import filter_proposals_patch, _get_top_n_idx_patch

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
    def test_rpn_head_anchor_generator_filter_proposal(self):

        rpn              = self.model.rpn
        rpn.filter_proposals = types.MethodType(filter_proposals_patch,
                                                rpn)
        rpn._get_top_n_idx   = types.MethodType(_get_top_n_idx_patch,
                                                rpn)
        
        rpn_head         = self.model.rpn.head
        anchor_generator = self.model.rpn.anchor_generator
        anchor_generator.forward = types.MethodType(anchor_generator_forward_patch,
                                                    anchor_generator)
        
        box_coder        = self.model.rpn.box_coder
        

        class HeadAnchorGenProxy(torch.nn.Module):

            def __init__(self,
                         rpn,
                         head,
                         anchor_gen,
                         box_coder):

                super(HeadAnchorGenProxy, self).__init__()
                self.rpn                      = rpn #for filter proposals
                self.head                     = head
                self.anchor_generator         = anchor_gen
                self.box_coder                = box_coder

            def forward(self,
                        images_tensors,
                        images_image_sizes,
                        features):

                features = list(features.values())
                
                objectness, pred_bbox_deltas = self.head(features)
                
                num_anchors_per_level = [o[0].numel() for o in objectness] # of anchors / feat level before concat
                
                from torch.onnx.operators import shape_as_tensor
                num_anchors_per_level_shape_tensors = [shape_as_tensor(o[0]) for o in objectness]
                num_anchors_per_level_fixed = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
                
                # print(num_anchors_per_level_shape_tensors)
                # num_anchors_per_level_fixed = [s.prod() for s in num_anchors_per_level_shape_tensors]
                # Could not find an implementation for the node ReduceProd(11)
                # print(num_anchors_per_level_fixed) # A list of tensors
                
                objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

                anchors  = self.anchor_generator(images_tensors,
                                                 images_image_sizes,
                                                 features)
                
                proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
                num_images = len(images_image_sizes)
                proposals = proposals.view(num_images, -1, 4)

                # PSX exporting debug
                boxes, scores = self.rpn.filter_proposals(proposals,
                                                          objectness,
                                                          images_image_sizes,
                                                          num_anchors_per_level_fixed)
                
                return boxes

        
        head_anchor_gen_mod = HeadAnchorGenProxy(rpn,
                                                 rpn_head,
                                                 anchor_generator,
                                                 box_coder)
        head_anchor_gen_mod.eval()

        features_g                       = self.im[0]["features"]
        transformed_images_tensors_g     = self.im[0]["transformed_images.tensors"]
        transformed_images_image_sizes_g = self.im[0]["transformed_images.image_sizes"] # list of tuple
        h, w = transformed_images_image_sizes_g[0]
        transformed_images_image_sizes_g = [torch.tensor([h, w])] #N = 1
        
        input_names  = ["image_list_tensors", "image_list_image_sizes",
                        "feat_0", "feat_1", "feat_2", "feat_3", "feat_pool"]
        
        # output_names = ["objs", "levels", "proposals"]
        # output_names = ["top_n_idx"]
        output_names = ["boxes"]
        
        dynamic_axes = { name : {2 : 'width', 3 : 'height'} for name in ["image_list_tensors",
                                                                         "feat_0",
                                                                         "feat_1",
                                                                         "feat_2",
                                                                         "feat_3",
                                                                         "feat_pool"]}
        # dynamic_axes.update({"top_n_idx"      : {1 : "idx"}})
        
        onnx_io = io.BytesIO()
        torch.onnx.export(head_anchor_gen_mod,
                          (transformed_images_tensors_g,
                           transformed_images_image_sizes_g, #loop is unrolled, don't need
                           features_g,),
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

            #rpn's output is proposal, check with torch ckpt dumped proposals directly
            proposals_g                      = im_b["proposals"]

            res_g = head_anchor_gen_mod(transformed_images_tensors_g,
                                        transformed_images_image_sizes_g,
                                        features_g)
            
            features_g                       = {k : to_numpy(v) for k, v in features_g.items()}            
            transformed_images_tensors_g     = to_numpy(transformed_images_tensors_g)
            transformed_images_image_sizes_g = to_numpy(transformed_images_image_sizes_g[0])
            res_g                            = [to_numpy(g) for g in res_g]
            proposals_g                      = [to_numpy(g) for g in proposals_g]
            
            res_ort \
                = ort_session.run(None,
                                  {
                                      "image_list_tensors"     : transformed_images_tensors_g,
                                      "image_list_image_sizes" : transformed_images_image_sizes_g,
                                      "feat_0"      : features_g["0"],
                                      "feat_1"      : features_g["1"],
                                      "feat_2"      : features_g["2"],
                                      "feat_3"      : features_g["3"],
                                      "feat_pool"   : features_g["pool"]
                                  })
            print("===Compare the onnx with its source mod===")
            for g, ort in zip(res_g, res_ort):
                self.assertEqual(g.shape, ort.shape)
                print("Tensor size check ...ok!", g.shape, ort.shape)
                try:
                    torch.testing.assert_allclose(g, ort , rtol=1e-03, atol=1e-04)
                except AssertionError as e:
                    print("Tensor allclose failed", g.shape)
                    print(str(e))
                else:
                    print("Tensor close check ...ok!")

            print("===Compare with the original dump===")
            for g, ort in zip(proposals_g, res_ort):
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
