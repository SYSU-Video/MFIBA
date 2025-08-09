"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False


    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def f2img(self, f, b, c, x, y):
        #c=256-->3
        x_new = max(x, 40)
        y_new = max(y, 40)
        padd_f = torch.zeros(b, c, x_new, y_new).to('cuda')
        padd_f[:, :, :x, :y] = f
        #zhijiereshape
        # flatten = f_resize.reshape(-1)
        # extra = b * 3 * 4608 * 1280 - flatten.numel()
        # zero = torch.zeros(extra)
        # flatten = flatten.to('cpu')
        # zero = zero.to('cpu')
        # img_flatten = torch.cat((flatten, zero))
        # img = img_flatten.reshape(b, 3, 4608, 1280)
        # img = img.to('cuda')
        #pintu
        img = torch.zeros(b, 3, 10*x_new, 9*y_new).to('cuda')
        for i in range(86):
            if i == 85:
                patch = padd_f[:, i * 3, :, :]
                img[0:b, 0, x_new * (i % 10): x_new * (i % 10 + 1), y_new * (i // 10): y_new * (i // 10 + 1)] = patch
            else:
                patch = padd_f[:, i * 3:(i + 1) * 3, :, :]
                img[0:b, 0:3, x_new * (i % 10): x_new * (i % 10 + 1), y_new * (i // 10): y_new * (i // 10 + 1)] = patch

        return img #f_resize-->img

    def img2f(self, img, b, c, x, y):
        x_new = max(x, 40)
        y_new = max(y, 40)
        #reshape
        # img_flatten = img.reshape(-1)
        # flatten = img_flatten[:b*256*256*256]
        # f = flatten.reshape(b, 256, 256, 256)
        #pintu
        f = torch.zeros(b, c, x_new, y_new).to('cuda')
        for i in range(86):
            if i == 85:
                patch = img[0:b, 0, x_new * (i % 10): x_new * (i % 10 + 1), y_new * (i // 10): y_new * (i // 10 + 1)]
                f[:, i * 3, :, :] = patch
            else:
                patch = img[0:b, 0:3, x_new * (i % 10): x_new * (i % 10 + 1), y_new * (i // 10): y_new * (i // 10 + 1)]
                f[:, i * 3:(i + 1) * 3, :, :] = patch

        f_decode = f[:, :, :x, :y]#img-->f
        return f_decode


    def forward(self, images, id=None, Q_dict=None, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features_org = self.backbone(images.tensors)

        # layer noise--------------------------------------------------------------
        features = {}
        for k, f in features_org.items():
            b, c, x, y = f.size()
            # guiyihua
            ma = torch.max(f)
            mi = torch.min(f)
            # f = (f - mi) / (ma - mi)
            # f_img = self.f2img(f, b, c, x, y)
            # npimg = f_img.cpu().numpy()
            # npimg = npimg[0, :, :, :]
            # npimg = npimg.transpose(1, 2, 0)
            # plt.imsave('/home/ta/liujunle/sda2/ELIC/mask_features_for_elic/features/' + id + '-' + k + '.png', npimg)

            # all layer compress:-----------------------------------
            if k == 'pool':
                layer_name = '4'
            else:
                layer_name = k
            # f_nosie = plt.imread(
            #     '/home/ta/liujunle/sda2/ELIC/kp_features_for_elic/feature_decode_'+Q_dict[layer_name]+'/' + id + '-' + k + '.png') # feature_decode_'+Q_dict[layer_name]
            f_nosie = plt.imread(
                '/home/ta/liujunle/sda2/ELIC/features_for_elic/feature_decode_'+Q_dict[layer_name]+'/' + id + '-' + k + '.png')
            f_nosie = f_nosie.transpose(2, 0, 1)
            f_nosie = torch.from_numpy(f_nosie).cuda()
            f_nosie = f_nosie.unsqueeze(0)
            f_decode = self.img2f(f_nosie, b, c, x, y)

            # f_decode = self.img2f(f_img, b, c, x, y)
            # fanguiyihua
            f_hat = (f_decode * (ma - mi)) + mi
            features[k] = f_hat

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
