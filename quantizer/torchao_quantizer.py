import os
import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig


class QuantizationManager:
    def __init__(self, device):
        self.device = device

    def quantize(self, model):
        quantize_(
            model,
            Int8WeightOnlyConfig()
        )
        return model

    def save(self, quantized_model, save_path="model_int8.pth"):
        torch.save(quantized_model.state_dict(), save_path)
        if os.path.exists(save_path):
            print(f"[INFO] Model saved to {save_path}")

    def load(self, model, load_path="model_int8.pth"):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"[ERROR] File not found: {load_path}")
            
        # Change model layout first before loading weights
        quantize_(
            model,
            Int8WeightOnlyConfig()
        )
        
        # Load the pre-quantized checkpoint
        device = next(model.parameters()).device
        model.load_state_dict(torch.load(load_path, map_location=device), assign=True)
        model.eval()
        return model
