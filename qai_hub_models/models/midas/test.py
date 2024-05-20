# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.models.midas.app import MidasApp
from qai_hub_models.models.midas.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.midas.demo import main as demo_main
from qai_hub_models.models.midas.model import MODEL_ASSET_VERSION, MODEL_ID, Midas
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "midas_output.png"
)


# Verify that the output from Torch is as expected.
@skip_clone_repo_check
def test_task():
    (_, _, height, width) = Midas.get_input_spec()["image"][0]
    app = MidasApp(Midas.from_pretrained(), height, width)
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = app.estimate_depth(original_image)
    output_image_oracle = load_image(OUTPUT_IMAGE_ADDRESS)

    np.testing.assert_allclose(
        np.asarray(output_image), np.asarray(output_image_oracle), atol=3
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    (_, _, height, width) = Midas.get_input_spec()["image"][0]
    traced_model = Midas.from_pretrained().convert_to_torchscript(check_trace=False)
    app = MidasApp(traced_model, height, width)
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = app.estimate_depth(original_image)
    output_image_oracle = load_image(OUTPUT_IMAGE_ADDRESS)

    np.testing.assert_allclose(
        np.asarray(output_image), np.asarray(output_image_oracle), atol=3
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
