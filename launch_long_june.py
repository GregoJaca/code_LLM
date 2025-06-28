import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil  

start_time = time.time()
cpu_start = psutil.cpu_percent()
ram_start = psutil.virtual_memory().used

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def compress_similarity_matrix(matrix, threshold=0.5, resolution=0.05):
    """Compress similarity matrix by quantizing and sparsifying"""
    matrix_cpu = matrix.cpu()
    mask = matrix_cpu.abs() >= threshold
    
    quantized = torch.round(matrix_cpu / resolution) * resolution
    quantized = quantized * mask.float()
    
    sparse_indices = torch.nonzero(quantized, as_tuple=False)
    sparse_values = quantized[quantized != 0]
    
    return {
        'indices': sparse_indices.numpy().astype('uint32'),
        'values': sparse_values.numpy().astype('float16'),
        'shape': matrix_cpu.shape
    }

def batched_cosine_similarity(tensor, batch_size=1024):
    """Calculate cosine similarity in batches to manage GPU memory"""
    n = tensor.size(0)
    tensor_norm = F.normalize(tensor, p=2, dim=1)
    
    similarity_parts = []
    
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch_i = tensor_norm[i:end_i]
        
        batch_similarities = []
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            batch_j = tensor_norm[j:end_j]
            
            sim_chunk = torch.mm(batch_i, batch_j.t())
            batch_similarities.append(sim_chunk)
        
        similarity_parts.append(torch.cat(batch_similarities, dim=1))
    
    return torch.cat(similarity_parts, dim=0)

def process_prompt(prompt, model, tokenizer, device, max_length, result_dir):
    ensure_dir(result_dir)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generation_config = {
            "max_new_tokens": max_length,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": None,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }
        outputs = model.generate(
            **inputs,
            **generation_config,
        )

    generated_tokens = outputs.sequences[0].tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # outputs.hidden_states: tuple of (num_steps, batch, seq, hidden)
    # Stack hidden states for each step
    hidden_states_steps = outputs.hidden_states  # tuple: (num_layers+1, num_steps, batch, seq, hidden)
    # For decoder-only models, hidden_states_steps is (num_steps, num_layers+1, batch, hidden)
    # We'll transpose to (num_layers+1, num_steps, batch, hidden)
    hidden_states_steps = [torch.stack([step[i][0] for step in outputs.hidden_states], dim=0) for i in range(len(outputs.hidden_states[0]))]
    # hidden_states_steps[i]: (num_steps, seq, hidden)

    # Last layer
    last_layer_idx = len(hidden_states_steps) - 1
    last_layer_states = hidden_states_steps[last_layer_idx]  # (num_steps, seq, hidden)
    last_layer_states = last_layer_states.reshape(-1, last_layer_states.size(-1))  # (seq_len, hidden)
    last_layer_states_cpu = last_layer_states.cpu().to(torch.float16)
    torch.save(last_layer_states_cpu, os.path.join(result_dir, "hidden_states_last.pt"))
    del last_layer_states
    torch.cuda.empty_cache()

    # Cosine similarity for last layer
    cosine_sim_last = batched_cosine_similarity(last_layer_states_cpu, batch_size=256)
    compressed_sim_last = compress_similarity_matrix(cosine_sim_last)
    torch.save(compressed_sim_last, os.path.join(result_dir, "cosine_sim_last_compressed.pt"))
    del cosine_sim_last
    torch.cuda.empty_cache()

    # First layer (token embedding)
    first_layer_states = hidden_states_steps[0]  # (num_steps, seq, hidden)
    first_layer_states = first_layer_states.reshape(-1, first_layer_states.size(-1))
    first_layer_states_cpu = first_layer_states.cpu().to(torch.float16)
    torch.save(first_layer_states_cpu, os.path.join(result_dir, "hidden_states_first.pt"))
    del first_layer_states
    torch.cuda.empty_cache()

    # Cosine similarity for first layer
    cosine_sim_first = batched_cosine_similarity(first_layer_states_cpu, batch_size=256)
    compressed_sim_first = compress_similarity_matrix(cosine_sim_first)
    torch.save(compressed_sim_first, os.path.join(result_dir, "cosine_sim_first_compressed.pt"))
    del cosine_sim_first
    torch.cuda.empty_cache()

    result_data = {
        "prompt": prompt,
        "model_configuration": {
            "model_name": model.config._name_or_path,
            "temperature": TEMPERATURE,
            "do_sample": False,
            "device": device,
            "max_new_tokens": max_length
        },
        "generated_text": generated_text,
        "generated_token_ids": generated_tokens
    }

    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"Results saved to {result_dir}")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 65536 # 100000
TEMPERATURE = 0

print(f"Using device: {DEVICE}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32 if DEVICE == "cuda" else torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model: {MODEL_NAME}")
model.to(DEVICE)
model.eval()

prompts = [
    "Examine how childhood experiences shape personality development. Discuss various influences including family environment, education, friendships, and significant life events. Explain psychological concepts like attachment theory and nature vs. nurture in accessible terms. Provide examples of how positive and negative experiences can affect adult personality traits and behaviors. Think step by step and reason exhaustively before you answer, finally give a comprehensive and long asnwer.",

]
prompt_names = [
    "childhood_personality_development",    # Psychology (Exploratory Tone)
]

main_results_dir = "long_run"
ensure_dir(main_results_dir)

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

