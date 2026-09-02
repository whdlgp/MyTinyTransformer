import os
import torch

from quantizer.torchao_quantizer import QuantizationManager
from block.model import MODEL_REGISTRY
from tokenizer.simple_tokenizer import Tokenizer
from dataloader.tinyshakespeare import prepare_files
from util.checkpoint import load_checkpoint
from util.config import load_config
from util.helper import get_device


def main():
    config = load_config()
    m_cfg = config["model"]
    test_cfg = config["test"]
    quant_cfg = config["quant"]
    device = get_device()
    
    # Download or load dataset
    train_path, _, _ = prepare_files()

    # Tokenizer (rebuilt from train.txt, same as training)
    with open(train_path, "r", encoding="utf-8") as f:
        tokenizer = Tokenizer.from_text(f.read())
    
    # Model
    model_cls = MODEL_REGISTRY[m_cfg["type"]]
    original_model = model_cls(
        tokenizer.vocab_size, m_cfg["d_model"], m_cfg["num_heads"],
        m_cfg["d_ff"], m_cfg["num_layers"], m_cfg["max_seq_len"], m_cfg["dropout"]
    ).to(device)
    
    # Load trained checkpoint
    checkpoint_path = test_cfg["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {test_cfg['checkpoint_path']}")
    load_checkpoint(checkpoint_path, original_model, device)
    original_model.eval()

    # Quantizer
    manager = QuantizationManager(device=device)
    
    # Do quantize
    quant_model = manager.quantize(original_model)

    # Save quantized model
    quant_checkpoint_path = quant_cfg["checkpoint_path"]
    manager.save(quant_model, quant_checkpoint_path)
    
    # Make model for quantized weights
    blank_model = model_cls(
        tokenizer.vocab_size, m_cfg["d_model"], m_cfg["num_heads"],
        m_cfg["d_ff"], m_cfg["num_layers"], m_cfg["max_seq_len"], m_cfg["dropout"]
    ).to(device)
    
    # Load quantized model
    loaded_model = manager.load(blank_model, quant_checkpoint_path)
    
    # Simple Test
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 10), dtype=torch.long, device=device)
    with torch.no_grad():
        output = loaded_model(dummy_input)
        
    print(f"[TEST] Output logits shape: {output.logits.shape}")


if __name__ == "__main__":
    main()
