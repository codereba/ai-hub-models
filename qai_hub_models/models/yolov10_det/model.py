# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models._shared.yolo.model import yolo_detect_postprocess
from qai_hub_models.utils.asset_loaders import (
    SourceAsRoot,
    find_replace_in_repo,
    wipe_sys_modules,
)
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ASSET_VERSION = 2
MODEL_ID = __name__.split(".")[-2]
SOURCE_REPO = "https://github.com/ultralytics/ultralytics"
SOURCE_REPO_COMMIT = "25307552100e4c03c8fec7b0f7286b4244018e15"

SUPPORTED_WEIGHTS = [
    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10m.pt",
    "yolov10l.pt",
    "yolov10x.pt",
]
DEFAULT_WEIGHTS = "yolov10n.pt"


class YoloV10Detector(BaseModel):
    """Exportable Yolo10 bounding box detector, end-to-end."""

    def __init__(
        self,
        model: nn.Module,
        include_postprocessing: bool = False,
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
            # Boxes and scores have different scales, so return separately
            find_replace_in_repo(
                repo_path,
                "ultralytics/nn/modules/head.py",
                "return torch.cat((dbox, cls.sigmoid()), 1)",
                "return (dbox, cls.sigmoid())",
            )
            find_replace_in_repo(
                repo_path,
                "ultralytics/nn/modules/head.py",
                "self.end2end",
                "False",
            )

            import ultralytics

            wipe_sys_modules(ultralytics)
            from ultralytics import YOLO as ultralytics_YOLO

            model = ultralytics_YOLO(ckpt_name).model
            assert isinstance(model, torch.nn.Module)

            return cls(
                model,
                include_postprocessing,
                split_output,
                use_quantized_postprocessing,
            )

    def forward(self, image):
        """
        Run YoloV10 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

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

        (boxes, scores), _ = self.model(image)
        if not self.include_postprocessing:
            if self.split_output:
                return boxes, scores
            return torch.cat([boxes, scores], dim=1)

        boxes, scores, classes = yolo_detect_postprocess(
            boxes, scores, self.use_quantized_postprocessing
        )

        return boxes, scores, classes

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 640,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names(
        include_postprocessing: bool = True, split_output: bool = False
    ) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "classes"]
        if split_output:
            return ["boxes", "scores"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(
            self.include_postprocessing, self.split_output
        )

    def get_evaluator(self) -> BaseEvaluator:
        return DetectionEvaluator(*self.get_input_spec()["image"][0][2:])

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


def get_hub_profile_options(
    self, target_runtime: TargetRuntime, other_profile_options: str = ""
) -> str:
    """
    Accuracy on ONNX & QNN runtime is not regained in NPU
    Issue: https://github.com/qcom-ai-hub/tetracode/issues/13108

    """
    profile_options = super().get_hub_profile_options(
        target_runtime, other_profile_options
    )
    if (
        target_runtime == TargetRuntime.ONNX  # unable to regain the accuracy
        or target_runtime == TargetRuntime.QNN  # unable to regain the accuracy
        and "--compute_unit" not in profile_options
    ):
        profile_options = profile_options + " --compute_unit cpu"
    return profile_options