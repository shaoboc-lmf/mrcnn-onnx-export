import torch
import torchvision

from torchvision.ops import boxes as box_ops

from torch.jit.annotations import List, Optional, Dict, Tuple

def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int)
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor])
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def anchor_generator_forward_patch(self,
                                   image_list_tensors,
                                   image_list_image_sizes,
                                   feature_maps):

    if torchvision._is_tracing():
        from torch.onnx import operators
        grid_sizes = list([operators.shape_as_tensor(feature_map)[-2:] for feature_map in feature_maps])
        image_size = operators.shape_as_tensor(image_list_tensors)[-2:]
        strides =  [ image_size / g for g in grid_sizes ]
    else:
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list_tensors.shape[-2:]
        strides = [[int(image_size[0] / g[0]), int(image_size[1] / g[1])] for g in grid_sizes]
    # TracerWarning: Converting a tensor to a Python integer
    
    dtype, device = feature_maps[0].dtype, feature_maps[0].device
    self.set_cell_anchors(dtype, device)
    # return self.cell_anchors
    
    # Ignore cache first because when we exporting, we only run one batch
    # anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
    anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
    # return anchors_over_all_feature_maps    
    
    anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
    # for i, (image_height, image_width) in enumerate(image_list.image_sizes):
    # num of images is constant?? loop over a dimension, N, so N can not be dynamic dimension
    for hw in image_list_image_sizes:
        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            anchors_in_image.append(anchors_per_feature_map)
        anchors.append(anchors_in_image)
    anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
    return anchors

def clip_boxes_to_image_patch(boxes, size):
    # type: (Tensor, Tuple[int, int])
    """
    Clip boxes so that they lie inside an image of size `size`.

    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """

    print("patch:clip_boxes_to_image")
    
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size
    
    # psx, the clamp is not traced correctly, min max are traced as constants
    # boxes_x = boxes_x.clamp(min=0, max=width)
    # boxes_y = boxes_y.clamp(min=0, max=height)
    # Use torch.min, torch.max to WAR

    height = height.to(torch.float32)
    width  = width.to(torch.float32)
    
    boxes_x = torch.max(boxes_x, torch.tensor(0.))
    boxes_x = torch.min(boxes_x, width)

    boxes_y = torch.max(boxes_y, torch.tensor(0.))
    boxes_y = torch.min(boxes_y, height)
    
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)

@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    # TODO : remove cast to IntTensor/num_anchors.dtype when
    #        ONNX Runtime version is updated with ReduceMin int64 support
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0).to(torch.int32)).to(num_anchors.dtype)

    return num_anchors, pre_nms_top_n

@torch.no_grad()
def _get_top_n_idx_patch(self, objectness, num_anchors_per_level):
    # type: (Tensor, List[int])

    print("patch:_get_top_n_idx")
    
    r = []
    offset = 0
    
    # PSX exporting debug
    start_list = [torch.tensor(0)]
    end_list   = [num_anchors_per_level[0].clone()]
    for cnt in num_anchors_per_level[1:]:
        start_list.append(end_list[-1].clone())
        end_list.append(end_list[-1] + cnt)
    objectness_list = [objectness[:,s:e] for s, e in zip(start_list, end_list)]
        
    # PSX exporting debug
    # for ob in objectness.split(num_anchors_per_level, 1):
    for ob in objectness_list:
        if torchvision._is_tracing():
            num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
        else:
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
        _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        r.append(top_n_idx + offset)
        offset += num_anchors
        
    return torch.cat(r, dim=1)
    
def filter_proposals_patch(self, proposals, objectness,  image_shapes, num_anchors_per_level):
    # type: (Tensor, Tensor, List[Tuple[int, int]], List[int])
    print("patch:filter_proposals")
    
    num_images = proposals.shape[0]
    device = proposals.device
    objectness = objectness.detach()
    objectness = objectness.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device)
        for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)

    top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level) #All call the patch

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]

    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
        
        # boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
        boxes = clip_boxes_to_image_patch(boxes, img_shape)
        
        keep = box_ops.remove_small_boxes(boxes, self.min_size)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[:self.post_nms_top_n()]
        boxes, scores = boxes[keep], scores[keep]
        final_boxes.append(boxes)
        final_scores.append(scores)

    return final_boxes, final_scores
