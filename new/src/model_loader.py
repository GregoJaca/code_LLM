import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(model_name, local_dir, device):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Loading model from local directory: {local_dir}")
    model.to(device)
    model.eval()
    return model, tokenizer
