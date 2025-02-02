# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models.deepseekr1_quantized.demo import deepseekr1_quantized_chat_demo


@pytest.mark.skip("#105 move slow_cloud and slow tests to nightly.")
@pytest.mark.slow_cloud
def test_demo():
    # Run demo and verify it does not crash
    deepseekr1_quantized_chat_demo(is_test=True)
