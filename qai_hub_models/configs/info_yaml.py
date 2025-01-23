# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import requests
from qai_hub.util.session import create_session

from qai_hub_models.configs._info_yaml_enums import (
    HF_AVAILABLE_LICENSES,
    MODEL_DOMAIN,
    MODEL_STATUS,
    MODEL_TAG,
    MODEL_USE_CASE,
)
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION, LLMDetails
from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, QAIHM_WEB_ASSET, load_yaml
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    MODELS_PACKAGE_NAME,
    QAIHM_MODELS_ROOT,
    QAIHM_PACKAGE_NAME,
    QAIHM_PACKAGE_ROOT,
    _get_qaihm_models_root,
)

__all__ = [
    "QAIHMModelInfo",
    "MODEL_DOMAIN",
    "MODEL_STATUS",
    "MODEL_TAG",
    "MODEL_USE_CASE",
]


@dataclass
class QAIHMModelInfo(BaseQAIHMConfig):
    """
    Schema & loader for model info.yaml.
    """

    # Name of the model as it will appear on the website.
    # Should have dashes instead of underscores and all
    # words capitalized. For example, `Whisper-Base-En`.
    name: str

    # Name of the model's folder within the repo.
    id: str

    # Whether or not the model is published on the website.
    # This should be set to public unless the model has poor accuracy/perf.
    status: MODEL_STATUS

    # Device form factors for which we don't publish performance data.
    private_perf_form_factors: list[ScorecardDevice.FormFactor]

    # A brief catchy headline explaining what the model does and why it may be interesting
    headline: str

    # The domain the model is used in such as computer vision, audio, etc.
    domain: MODEL_DOMAIN

    # A 2-3 sentence description of how the model can be used.
    description: str

    # What task the model is used to solve, such as object detection, classification, etc.
    use_case: MODEL_USE_CASE

    # A list of applicable tags to add to the model
    tags: list[MODEL_TAG]

    # A list of real-world applicaitons for which this model could be used.
    # This is free-from and almost anything reasonable here is fine.
    applicable_scenarios: list[str]

    # A list of other similar models in the repo.
    # Typically, any model that performs the same task is fine.
    # If nothing fits, this can be left blank. Limit to 3 models.
    related_models: list[str]

    # A list of device types for which this model could be useful.
    # If unsure what to put here, default to `Phone` and `Tablet`.
    form_factors: list[ScorecardDevice.FormFactor]

    # Whether the model has a static image uploaded in S3. All public models must have this.
    has_static_banner: bool

    # Whether the model has an animated asset uploaded in S3. This is optional.
    has_animated_banner: bool

    # CodeGen options from code-gen.yaml in the model's folder.
    code_gen_config: QAIHMModelCodeGen

    # A list of datasets for which the model has pre-trained checkpoints
    # available as options in `model.py`. Typically only has one entry.
    dataset: list[str]

    # A list of a few technical details about the model.
    #   Model checkpoint: The name of the downloaded model checkpoint file.
    #   Input resolution: The size of the model's input. For example, `2048x1024`.
    #   Number of parameters: The number of parameters in the model.
    #   Model size: The file size of the downloaded model asset.
    #       This and `Number of parameters` should be auto-generated by running `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_name>`
    #   Number of output classes: The number of classes the model can classify or annotate.
    technical_details: dict[str, str]

    # The license type of the original model repo.
    license_type: str

    # Some models are made by company
    model_maker_id: Optional[str] = None

    # Link to the research paper where the model was first published. Usually an arxiv link.
    research_paper: Optional[str] = None

    # The title of the research paper.
    research_paper_title: Optional[str] = None

    # A link to the original github repo with the model's code.
    source_repo: Optional[str] = None

    # A link to the model's license. Most commonly found in the github repo it was cloned from.
    license: Optional[str] = None

    # Whether the model is compatible with the IMSDK Plugin for IOT devices
    imsdk_supported: bool = False

    # A link to the AIHub license, unless the license is more restrictive like GPL.
    # In that case, this should point to the same as the model license.
    deploy_license: Optional[str] = None

    # Should be set to `AI Model Hub License`, unless the license is more restrictive like GPL.
    # In that case, this should be the same as the model license.
    deploy_license_type: Optional[str] = None

    # If set, model assets shouldn't distributed.
    restrict_model_sharing: bool = False

    # If status is private, this must have a reference to an internal issue with an explanation.
    status_reason: Optional[str] = None

    # If the model outputs class indices, this field should be set and point
    # to a file in `qai_hub_models/labels`, which specifies the name for each index.
    labels_file: Optional[str] = None

    # It is a large language model (LLM) or not.
    model_type_llm: bool = False

    # Add per device, download, app and if the model is available for purchase.
    llm_details: Optional[LLMDetails] = None

    @property
    def is_quantized(self) -> bool:
        return (
            self.code_gen_config.is_aimet or self.code_gen_config.use_hub_quantization
        )

    def validate(self) -> Optional[str]:
        """Returns false with a reason if the info spec for this model is not valid."""
        # Validate ID
        if self.id not in MODEL_IDS:
            return f"{self.id} is not a valid QAI Hub Models ID."
        if " " in self.id or "-" in self.id:
            return "Model IDs cannot contain spaces or dashes."
        if self.id.lower() != self.id:
            return "Model IDs must be lowercase."

        # Validate (used as repo name for HF as well)
        if " " in self.name:
            return "Model Name must not have a space."

        # Headline should end with period
        if not self.headline.endswith("."):
            return "Model headlines must end with a period."

        # Quantized models must contain quantized tag
        if ("quantized" in self.id) and (MODEL_TAG.QUANTIZED not in self.tags):
            return f"Quantized models must have quantized tag. tags: {self.tags}"
        if ("quantized" not in self.id) and (MODEL_TAG.QUANTIZED in self.tags):
            return f"Models with a quantized tag must have 'quantized' in the id. tags: {self.tags}"

        # Validate related models are present
        for r_model in self.related_models:
            if r_model not in MODEL_IDS:
                return f"Related model {r_model} is not a valid model ID."
            if r_model == self.id:
                return f"Model {r_model} cannot be related to itself."

        # If paper is arxiv, it should be an abs link
        if self.research_paper is not None and self.research_paper.startswith(
            "https://arxiv.org/"
        ):
            if "/abs/" not in self.research_paper:
                return "Arxiv links should be `abs` links, not link directly to pdfs."

        # If license_type does not match the map, return an error
        if self.license_type not in HF_AVAILABLE_LICENSES:
            return f"license can be one of these: {HF_AVAILABLE_LICENSES}"

        # Status Reason
        if self.status == MODEL_STATUS.PRIVATE and not self.status_reason:
            return "Private models must set `status_reason` in info.yaml with a link to the related issue."

        if self.status == MODEL_STATUS.PUBLIC and self.status_reason:
            return "`status_reason` in info.yaml should not be set for public models."

        # Labels file
        if self.labels_file is not None:
            if not os.path.exists(ASSET_CONFIG.get_labels_file_path(self.labels_file)):
                return f"Invalid labels file: {self.labels_file}"

        # Required assets exist
        if self.status == MODEL_STATUS.PUBLIC:
            if not os.path.exists(self.get_package_path() / "info.yaml"):
                return "All public models must have an info.yaml"

            if (
                (
                    self.code_gen_config.tflite_export_failure_reason
                    or self.code_gen_config.tflite_export_disable_reason
                )
                and (
                    self.code_gen_config.qnn_export_failure_reason
                    or self.code_gen_config.qnn_export_disable_reason
                )
                and (
                    self.code_gen_config.onnx_export_failure_reason
                    or self.code_gen_config.onnx_export_disable_reason
                )
            ):
                return "Public models must support at least one export path"

            if not self.has_static_banner:
                return "Public models must have a static asset."

        session = create_session()
        if self.has_static_banner:
            static_banner_url = ASSET_CONFIG.get_web_asset_url(
                self.id, QAIHM_WEB_ASSET.STATIC_IMG
            )
            if session.head(static_banner_url).status_code != requests.codes.ok:
                return f"Static banner is missing at {static_banner_url}"
        if self.has_animated_banner:
            animated_banner_url = ASSET_CONFIG.get_web_asset_url(
                self.id, QAIHM_WEB_ASSET.ANIMATED_MOV
            )
            if session.head(animated_banner_url).status_code != requests.codes.ok:
                return f"Animated banner is missing at {animated_banner_url}"

        expected_qaihm_repo = Path("qai_hub_models") / "models" / self.id
        if expected_qaihm_repo != ASSET_CONFIG.get_qaihm_repo(self.id):
            return "QAIHM repo not pointing to expected relative path"

        expected_example_use = f"qai_hub_models/models/{self.id}#example--usage"
        if expected_example_use != ASSET_CONFIG.get_example_use(self.id):
            return "Example-usage field not pointing to expected relative path"

        model_is_available = self.status == MODEL_STATUS.PUBLIC

        # Check that model_type_llm and llm_details fields
        if self.model_type_llm:
            if not self.llm_details:
                return "llm_details must be set if model type is LLM"

            if bad_llm_details := self.llm_details.validate():
                return bad_llm_details

            model_is_available = self.llm_details.call_to_action not in [
                LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE,
                LLM_CALL_TO_ACTION.COMING_SOON,
                LLM_CALL_TO_ACTION.CONTACT_US,
            ]

            # Download URL can only be validated in a scope with a model ID, so this
            # is validated here rather than on the LLM details class' validator.
            if self.llm_details.devices:
                for device_runtime_config_mapping in self.llm_details.devices.values():
                    for runtime_detail in device_runtime_config_mapping.values():
                        version = runtime_detail.model_download_url.split("/")[0][1:]
                        relative_path = "/".join(
                            runtime_detail.model_download_url.split("/")[1:]
                        )
                        model_download_url = ASSET_CONFIG.get_model_asset_url(
                            self.id, version, relative_path
                        )
                        if (
                            session.head(model_download_url).status_code
                            != requests.codes.ok
                        ):
                            return f"Download URL is missing at {runtime_detail.model_download_url}"
        elif self.llm_details:
            return "Model type must be LLM if llm_details is set"

        if not self.deploy_license and model_is_available:
            return "deploy_license cannot be empty"

        if not self.deploy_license_type and model_is_available:
            return "deploy_license_type cannot be empty"

        # Check code gen
        return self.code_gen_config.validate()

    def get_package_name(self):
        return f"{QAIHM_PACKAGE_NAME}.{MODELS_PACKAGE_NAME}.{self.id}"

    def get_package_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return _get_qaihm_models_root(root) / self.id

    def get_model_definition_path(self):
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "model.py"
        )

    def get_demo_path(self):
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "demo.py"
        )

    def get_labels_file_path(self):
        if self.labels_file is None:
            return None
        return ASSET_CONFIG.get_labels_file_path(self.labels_file)

    def get_info_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "info.yaml"

    def get_hf_pipeline_tag(self):
        return self.use_case.map_to_hf_pipeline_tag()

    def get_hugging_face_metadata(self, root: Path = QAIHM_PACKAGE_ROOT):
        # Get the metadata for huggingface model cards.
        hf_metadata: dict[str, Union[str, list[str]]] = dict()
        hf_metadata["library_name"] = "pytorch"
        hf_metadata["license"] = self.license_type
        hf_metadata["tags"] = [tag.name.lower() for tag in self.tags] + ["android"]
        hf_metadata["pipeline_tag"] = self.get_hf_pipeline_tag()
        return hf_metadata

    def get_model_details(self):
        # Model details.
        details = (
            "- **Model Type:** "
            + self.use_case.__str__().lower().capitalize()
            + "\n- **Model Stats:**"
        )
        for name, val in self.technical_details.items():
            details += f"\n  - {name}: {val}"
        return details

    def get_perf_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "perf.yaml"

    def get_code_gen_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "code-gen.yaml"

    def get_readme_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "README.md"

    def get_hf_model_card_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "HF_MODEL_CARD.md"

    def get_requirements_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "requirements.txt"

    def has_model_requirements(self, root: Path = QAIHM_PACKAGE_ROOT):
        return os.path.exists(self.get_requirements_path(root))

    @property
    def is_gen_ai_model(self) -> bool:
        return MODEL_TAG.LLM in self.tags or MODEL_TAG.GENERATIVE_AI in self.tags

    @classmethod
    def from_model(cls: type[QAIHMModelInfo], model_id: str) -> QAIHMModelInfo:
        schema_path = QAIHM_MODELS_ROOT / model_id / "info.yaml"
        code_gen_path = QAIHM_MODELS_ROOT / model_id / "code-gen.yaml"
        if not os.path.exists(schema_path):
            raise ValueError(f"{model_id} does not exist")
        return cls.from_yaml_and_code_gen(schema_path, code_gen_path)

    @classmethod
    def from_dict(
        cls: type[QAIHMModelInfo], val_dict: dict[str, Any]
    ) -> QAIHMModelInfo:
        val_dict["status"] = MODEL_STATUS.from_string(val_dict["status"])
        val_dict["private_perf_form_factors"] = [
            ScorecardDevice.FormFactor.from_string(tag)
            for tag in val_dict.get("private_perf_form_factors", [])
        ]
        val_dict["domain"] = MODEL_DOMAIN.from_string(val_dict["domain"])
        val_dict["use_case"] = MODEL_USE_CASE.from_string(val_dict["use_case"])
        val_dict["tags"] = [MODEL_TAG.from_string(tag) for tag in val_dict["tags"]]
        val_dict["form_factors"] = [
            ScorecardDevice.FormFactor.from_string(tag)
            for tag in val_dict["form_factors"]
        ]
        if llm_details := val_dict.get("llm_details", None):
            val_dict["llm_details"] = LLMDetails.from_dict(llm_details)

        code_gen_config = val_dict.get("code_gen_config", None)
        if isinstance(code_gen_config, dict):
            val_dict["code_gen_config"] = QAIHMModelCodeGen.from_dict(code_gen_config)
        elif not isinstance(code_gen_config, QAIHMModelCodeGen):
            raise ValueError("code_gen_config must be dict or QAIHMModelCodeGen")

        return super().from_dict(val_dict)

    @classmethod
    def from_yaml_and_code_gen(
        cls: type[QAIHMModelInfo], info_path: str | Path, code_gen_path: str | Path
    ) -> QAIHMModelInfo:
        info = load_yaml(info_path)
        info["code_gen_config"] = QAIHMModelCodeGen.from_yaml(code_gen_path)
        return cls.from_dict(info)
