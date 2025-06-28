import os
import json
import torch
import torch.nn.functional as F
# import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil  


start_time = time.time()
cpu_start = psutil.cpu_percent()
ram_start = psutil.virtual_memory().used






def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_prompt(prompt, model, tokenizer, device, max_length, result_dir):
    """Process a single prompt and save results"""
    ensure_dir(result_dir)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text with hidden states
    with torch.no_grad():

        generation_config = {
            "max_new_tokens": max_length,  # exactly 100000 tokens
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": None,  # disables EOS-based stopping
        }
        
        outputs = model.generate(
            **inputs,
            **generation_config,
            return_dict_in_generate=True,
        )
    
    # Extract generated text
    generated_tokens = outputs.sequences[0].tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Determine layer indices for first, middle, and last layers
    num_layers = len(outputs.hidden_states[0])
    selected_layers = {
        "last": num_layers - 1
    }
    
    # extract only the hidden states we need (first, middle, last layer)
    layer_hidden_states = {}
    for layer_name, layer_idx in selected_layers.items():
        # Extract the specified layer's hidden states across all generation steps
        # and concatenate them into a single tensor
        layer_states = torch.cat([
            step_hidden_states[layer_idx]
            for step_hidden_states in outputs.hidden_states
        ], dim=1)
        
        # Store as a reference to avoid copying
        layer_hidden_states[layer_name] = layer_states
        
        # Also save this layer's hidden states
        torch.save(
            layer_states, 
            os.path.join(result_dir, f"hidden_states_{layer_name}.pt")
        )
    
    
    # Calculate cosine similarity and L2 distance between each pair of layers
    for i, layer1_name in enumerate(selected_layers.keys()):
        for j, layer2_name in enumerate(selected_layers.keys()):
            # Only process when j >= i to avoid duplicate calculations
            if j >= i:

                # Get tensor references
                tensor1 = layer_hidden_states[layer1_name]
                tensor2 = layer_hidden_states[layer2_name]
                
                # Reshape tensors using view when possible to avoid copies
                tensor1_flat = tensor1.reshape(-1, tensor1.size(-1))
                tensor2_flat = tensor2.reshape(-1, tensor2.size(-1))
                
                # Calculate cosine similarity
                tensor1_norm = F.normalize(tensor1_flat, p=2, dim=1)
                tensor2_norm = F.normalize(tensor2_flat, p=2, dim=1)
                cosine_sim = torch.mm(tensor1_norm, tensor2_norm.t())
                
                # Store results
                matrix_name_cos = f"cosine_sim_{layer1_name}_{layer2_name}"

                # For similarity matrices:
                torch.save(
                    cosine_sim.detach().cpu(),
                    os.path.join(result_dir, f"{matrix_name_cos}.pt")
                )

                
    # Extract token-by-token output
    tokens_generated = []
    for i in range(len(generated_tokens)):
        token = generated_tokens[i]
        token_text = tokenizer.decode([token])
        tokens_generated.append(token_text)

    
    # Save results
    result_data = {
        "prompt": prompt,
        "model_configuration": {
            "model_name": model.config._name_or_path,
            "temperature": TEMPERATURE,
            "do_sample": False,
            "device": device,
            "max_new_tokens": max_length  # exactly 100000 tokens
        },
        "generated_text": generated_text,
    }
    
    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)
    
    # Save the tokens data
    with open(os.path.join(result_dir, "tokens.json"), "w") as f:
        json.dump(tokens_generated, f, indent=2)
    
    print(f"Results saved to {result_dir}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


# Model configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 100000
TEMPERATURE = 0


print(f"Using device: {DEVICE}")


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32 if DEVICE == "cuda" else torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model: {MODEL_NAME}")
# Move model to device and set eval mode (ONCE)
model.to(DEVICE)
model.eval()  # Disables dropout/batch norm if any

# List of prompts to process
prompts = [ 

]

prompt_names = [

]

# Create main results directory
main_results_dir = "long_run"
ensure_dir(main_results_dir)

# Process each prompt
for i, prompt in enumerate(prompts):
    prompt_dir = os.path.join(main_results_dir, prompt_names[i])
    process_prompt(prompt, model, tokenizer, DEVICE, MAX_LENGTH, prompt_dir)

print("All prompts processed successfully!")















end_time = time.time()
execution_time = end_time - start_time
print("\n\n=== PERFORMANCE METRICS ===\n")
print(f"Total execution time: {execution_time:.2f} seconds\n")
print(f"CPU usage: {psutil.cpu_percent()}%\n")
print(f"RAM used: {(psutil.virtual_memory().used - ram_start)/1024/1024:.2f} MB\n")
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.max_memory_allocated()/1024/1024:.2f} MB\n")
else:
    print("cuda not available at the end")













