import platform
import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"device: cuda ({props.name}, {props.total_memory / 1024 ** 3:.1f}GB)")
    else:
        device = torch.device("cpu")
        print(f"device: cpu ({platform.processor() or platform.machine()})")
    return device