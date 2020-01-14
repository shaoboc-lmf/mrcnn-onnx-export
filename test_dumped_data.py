import torch
import torchvision
import unittest

from .utils import *

class TestDumpedData(unittest.TestCase):

    def setUp(self):
        self.im    = torch.load("dumped_data/im_b5")
        self.model = torchvision.models.detection.__dict__["maskrcnn_resnet50_fpn"](num_classes=91,
                                                                                    pretrained=True)
        self.model.eval()

    @torch.no_grad()
    def test_transform(self):

        images_g                         = self.im["images"]
        transformed_images_tensors_g     = self.im["transformed_images.tensors"]
        transformed_images_image_sizes_g = self.im["transformed_images.image_sizes"]
        transformed_images_g             = ImageList(transformed_images_tensors_g,
                                                     transformed_images_image_sizes_g)
        transform = self.model.transform

        transformed_images, _ = transform(images_g, None)

        #test image
        torch.testing.assert_allclose(transformed_images.tensors,
                                      transformed_images_tensors_g,
                                      rtol=1e-03, atol=1e-05)

        #test image_size
        self.assertTrue(type(transformed_images.image_sizes)  == list)
        self.assertTrue(len(transformed_images_image_sizes_g) == len(transformed_images.image_sizes))

        for g, t in zip(transformed_images_image_sizes_g, transformed_images.image_sizes):

            g_h, g_w = g
            t_h, t_w = t

            self.assertEqual(g_h, t_h)
            self.assertEqual(g_w, t_w)

    @torch.no_grad()
    def test_backbone(self):

        transformed_images_tensors_g = self.im["transformed_images.tensors"]
        features_g                   = self.im["features"]

        backbone = self.model.backbone

        features = backbone(transformed_images_tensors_g)

        print("Checking OrderedDict")
        for (g_k, g_v), (t_k, t_v) in zip(features_g.items(), features.items()):
            print("...Checking {}".format(g_k))
            self.assertEqual(g_k, t_k)
            torch.testing.assert_allclose(g_v, t_v, rtol=1e-03, atol=1e-05)

    @torch.no_grad()
    def test_rpn(self):
        features_g                       = self.im["features"]
        transformed_images_tensors_g     = self.im["transformed_images.tensors"]
        transformed_images_image_sizes_g = self.im["transformed_images.image_sizes"]
        transformed_images_g             = ImageList(transformed_images_tensors_g,
                                                     transformed_images_image_sizes_g)
        proposals_g                      = self.im["proposals"]
        
        rpn = self.model.rpn

        proposals, _ = rpn(transformed_images_g, features_g, None)

        self.assertTrue(len(proposals) == 1 == len(proposals_g))
        torch.testing.assert_allclose(proposals[0],
                                      proposals_g[0],
                                      rtol=1e-03, atol=1e-05)
        
    @torch.no_grad()
    def test_roi_heads(self):
        features_g                       = self.im["features"]
        proposals_g                      = self.im["proposals"]
        transformed_images_image_sizes_g = self.im["transformed_images.image_sizes"]
        detections_g                     = self.im["detections"]

        roi_heads = self.model.roi_heads

        detections, _ = roi_heads(features_g, proposals_g, transformed_images_image_sizes_g, None)

        self.assertTrue(len(detections) == 1 == len(detections_g))
        print("Checking Dict")
        for (g_k, g_v), (t_k, t_v) in zip(detections_g[0].items(), detections[0].items()):
            print("...Checking {}".format(g_k))
            self.assertEqual(g_k, t_k)
            torch.testing.assert_allclose(g_v, t_v, rtol=1e-03, atol=1e-05)
    
    @torch.no_grad()
    def test_transform_postprocess(self):
        detections_g                     = self.im["detections"]
        transformed_images_image_sizes_g = self.im["transformed_images.image_sizes"]
        original_image_sizes_g           = self.im["original_image_sizes"]
        detections_postprocessed_g       = self.im["detections_postprocessed"]

        transform_post = self.model.transform.postprocess

        detections_postprocessed = transform_post(detections_g,
                                                  transformed_images_image_sizes_g,
                                                  original_image_sizes_g)

        self.assertTrue(len(detections_postprocessed) == 1 == len(detections_postprocessed_g))
        print("Checking Dict")
        for (g_k, g_v), (t_k, t_v) in zip(detections_postprocessed_g[0].items(),
                                          detections_postprocessed[0].items()):
            print("...Checking {}".format(g_k))
            self.assertEqual(g_k, t_k)
            torch.testing.assert_allclose(g_v, t_v, rtol=1e-03, atol=1e-05)

if __name__ == "__main__":
    unittest.main()
