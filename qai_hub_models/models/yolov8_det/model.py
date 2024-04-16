# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models._shared.yolo.utils import (
    box_transform_xywh2xyxy_split_input,
    transform_box_layout_xywh2xyxy,
)
from qai_hub_models.utils.asset_loaders import (
    SourceAsRoot,
    find_replace_in_repo,
    wipe_sys_modules,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]
SOURCE_REPO = "https://github.com/ultralytics/ultralytics"
SOURCE_REPO_COMMIT = "3208eb72ef277b0b825306a84df6c460a8406647"

SUPPORTED_WEIGHTS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]
DEFAULT_WEIGHTS = "yolov8n.pt"


class YoloV8Detector(BaseModel):
    """Exportable YoloV8 bounding box detector, end-to-end."""

    def __init__(
        self,
        model: nn.Module,
        include_postprocessing: bool = True,
        split_output: bool = False,
        use_quantized_postprocessing: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        self.use_quantized_postprocessing = use_quantized_postprocessing

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
        use_quantized_postprocessing: bool = False,
    ):
        with SourceAsRoot(
            SOURCE_REPO,
            SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            # Functionally equivalent re-writes that make it torch.fx.Graph compatible
            find_replace_in_repo(
                repo_path,
                "ultralytics/nn/modules/block.py",
                "softmax(1)",
                "softmax(dim=1)",
            )
            find_replace_in_repo(
                repo_path,
                "ultralytics/nn/modules/block.py",
                "y = list(self.cv1(x).chunk(2, 1))",
                "y = self.cv1(x).chunk(2, 1)\n        y = [y[0], y[1]]",
            )
            find_replace_in_repo(
                repo_path,
                "ultralytics/nn/modules/head.py",
                "self.dynamic or self.shape != shape",
                "False",
            )
            # Boxes and scores have different scales, so return separately
            find_replace_in_repo(
                repo_path,
                "ultralytics/nn/modules/head.py",
                "y = torch.cat((dbox, cls.sigmoid()), 1)",
                "return (dbox, cls.sigmoid())",
            )

            import ultralytics

            wipe_sys_modules(ultralytics)
            from ultralytics import YOLO as ultralytics_YOLO

            model = ultralytics_YOLO(ckpt_name).model
            model.eval()
            return cls(
                model,
                include_postprocessing,
                split_output,
                use_quantized_postprocessing,
            )

    def forward(self, image):
        """
        Run YoloV8 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                    Range: float[0, 1]
                    3-channel Color Space: RGB

        Returns:
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
                scores: torch.Tensor
                    class scores multiplied by confidence: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
            Elif self.split_output:
                boxes: torch.Tensor
                    Bounding box predictions in xywh format. Shape [batch, 4, num_preds].
                scores: torch.Tensor
                    Full score distribution over all classes for each box.
                    Shape [batch, num_classes, num_preds].
            Else:
                predictions: torch.Tensor
                Same as previous case but with boxes and scores concatenated into a single tensor.
                Shape [batch, 4 + num_classes, num_preds]
        """
        boxes, scores = self.model(image)
        if not self.include_postprocessing:
            if self.split_output:
                return boxes, scores
            return torch.cat([boxes, scores], dim=1)

        boxes, scores, classes = yolov8_detect_postprocess(
            boxes, scores, self.use_quantized_postprocessing
        )
        return boxes, scores, classes

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_channels: int = 3,
        height: int = 640,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, num_channels, height, width), "float32")}

    def get_evaluator(self) -> BaseEvaluator:
        return DetectionEvaluator(640, 640)


def yolov8_detect_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    use_quantized_postprocessing: bool = False,
):
    """
    Post processing to break YoloV8 detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores and classes.

    Parameters:
        detector_output: torch.Tensor
            The output of Yolo Detection model
            Shape is [batch, k, num_preds]
                Where, k = # of classes + 4
                The array dimension k is structured as follows:
                    [box coordintes, # of classes]
                where box coordinates are [x_center, y_center, w, h]
        use_quantized_postprocessing: bool
            If post-processing a non-quantized model, need to split the bounding box
            processing into multiple smaller tensors due to NPU limitations.
            If quantized, the entire processing can be done in a single operation.

    Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            class scores multiplied by confidence: Shape is [batch, num_preds]
        class_idx: torch.tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    boxes = torch.permute(boxes, [0, 2, 1])
    scores = torch.permute(scores, [0, 2, 1])

    # Convert boxes to (x1, y1, x2, y2)
    # Doing transform in fp16 requires special logic to keep on NPU
    if use_quantized_postprocessing:
        boxes = box_transform_xywh2xyxy_split_input(boxes[..., 0:2], boxes[..., 2:4])
    else:
        boxes = transform_box_layout_xywh2xyxy(boxes)

    # Get class ID of most likely score.
    scores, class_idx = torch.max(scores, -1, keepdim=False)

    return boxes, scores, class_idx
