import os

import numpy as np
import torch
import torchvision
from PIL import Image

import qai_hub as hub

# 1. Load pre-trained PyTorch model from torchvision
torch_model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()

# 2. Trace the model to TorchScript format
input_shape = (1, 3, 224, 224)
pt_model = torch.jit.trace(torch_model, torch.rand(input_shape))

# 3. Compile the model to ONNX
device = hub.Device("Samsung Galaxy S24 (Family)")
compile_onnx_job = hub.submit_compile_job(
    model=pt_model,
    device=device,
    input_specs=dict(image_tensor=input_shape),
    options="--target_runtime onnx",
)
assert isinstance(compile_onnx_job, hub.CompileJob)

unquantized_onnx_model = compile_onnx_job.get_target_model()
assert isinstance(unquantized_onnx_model, hub.Model)