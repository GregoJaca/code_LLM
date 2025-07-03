import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
import time
import psutil
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

start_time = time.time()
cpu_start = psutil.cpu_percent()
ram_start = psutil.virtual_memory().used

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def chunked_cosine_similarity_and_save(tensor, result_dir, batch_size=256*4, threshold=0.5, resolution=0.05, save_name="cosine_sim_matrix.pt"):
    # Squeeze batch dimension if present (e.g. [1, seq_len, hidden_dim] -> [seq_len, hidden_dim])
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    n = tensor.size(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = tensor.to(device)
    tensor_norm = F.normalize(tensor, p=2, dim=1)
    shape = (n, n)
    # Preallocate the full quantized matrix on CPU (float16 for space efficiency)
    full_quantized = torch.zeros((n, n), dtype=torch.float16, device='cpu')
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch_i = tensor_norm[i:end_i].to(device)
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            batch_j = tensor_norm[j:end_j].to(device)
            sim_chunk = torch.mm(batch_i, batch_j.t())
            mask = sim_chunk.abs() >= threshold
            quantized = torch.round(sim_chunk / resolution) * resolution
            quantized = quantized * mask.float()
            quantized_cpu = quantized.detach().cpu().to(torch.float16)
            full_quantized[i:end_i, j:end_j] = quantized_cpu
            del sim_chunk, mask, quantized, quantized_cpu
            torch.cuda.empty_cache()

    # I tried to use sparse tensor, but it used more memory!
    # full_quantized_sparse = full_quantized.to_sparse()
    # torch.save(full_quantized_sparse, os.path.join(result_dir, save_name))
    torch.save(full_quantized, os.path.join(result_dir, save_name))

    # meta = {'shape': shape, 'batch_size': batch_size, 'threshold': threshold, 'resolution': resolution}
    # with open(os.path.join(result_dir, "cosine_sim_matrix_meta.json"), "w") as f:
    #     json.dump(meta, f, indent=2)

def process_prompt(prompt, model, tokenizer, device, max_length, result_dir):
    ensure_dir(result_dir)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    num_layers = model.config.num_hidden_layers
    selected_layers = {"first": 0, "last": num_layers - 1}
    
    # Store hidden states on CPU to save GPU memory
    hidden_states_storage = {layer_name: [] for layer_name in selected_layers.keys()}
    
    generated_tokens = input_ids.squeeze().tolist()
    past_key_values = None
    
    SLIDING_WINDOW_STRIDE = 64 # For efficiency, slide window in chunks

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(max_length):
            # Prepare model inputs
            if past_key_values is not None:
                # If we have past_key_values, we only need the last token as input
                current_input_ids = input_ids[:, -1:]
                # Correctly handle position_ids for sliding window. The position of the current token is the length of the sequence so far.
                position_ids = torch.tensor([[len(generated_tokens) - 1]], device=device, dtype=torch.long)
            else:
                current_input_ids = input_ids
                position_ids = None

            model_inputs = {
                "input_ids": current_input_ids, 
                "past_key_values": past_key_values, 
                "use_cache": True, 
                "output_hidden_states": True,
                "position_ids": position_ids
            }

            # Forward pass
            outputs = model(**model_inputs)
            
            # Get the next token
            next_token_logits = outputs.logits[:, -1, :]
            # Apply temperature
            next_token_logits = next_token_logits / TEMPERATURE
            # Apply top-p sampling
            filtered_logits = top_p_filtering(next_token_logits, top_p=0.95)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            
            # Append the new token to our generated tokens list
            generated_tokens.append(next_token.item())
            
            # Update input_ids for the next iteration
            input_ids = next_token

            # Handle hidden states
            # outputs.hidden_states contains embeddings (idx 0) + one state per layer (idx 1 to num_layers)
            for layer_name, layer_idx in selected_layers.items():
                # "first" corresponds to the embedding layer (index 0)
                # "last" corresponds to the final layer's output (index num_layers)
                state_idx = 0 if layer_name == "first" else num_layers
                hidden_state = outputs.hidden_states[state_idx][:, -1, :].cpu().to(torch.float16)
                hidden_states_storage[layer_name].append(hidden_state)

            # Update past_key_values
            past_key_values = outputs.past_key_values

            # More efficient sliding window for KV cache
            if past_key_values is not None and past_key_values.get_seq_length() > ATTENTION_WINDOW_SIZE + SLIDING_WINDOW_STRIDE:
                new_cache = []
                for key, value in past_key_values: # Iterate through the layers in the cache
                    # Truncate the sequence length dimension
                    new_key = key[:, :, -ATTENTION_WINDOW_SIZE:]
                    new_value = value[:, :, -ATTENTION_WINDOW_SIZE:]
                    new_cache.append((new_key, new_value))
                past_key_values = DynamicCache.from_legacy_cache(past_key_values=tuple(new_cache))


            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # Decode the generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Save the tokens data
    tokens_generated_text = [tokenizer.decode([token]) for token in generated_tokens]
    with open(os.path.join(result_dir, "tokens.json"), "w") as f:
        json.dump(tokens_generated_text, f, indent=2)

    # Save the prompt and model configuration and answer text
    result_data = {
        "prompt": prompt,
        "model_configuration": {
            "model_name": model.config._name_or_path,
            "temperature": TEMPERATURE,
            "top_p": 0.95,
            "do_sample": True,
            "repetition_penalty":1.1,
            "device": device,
            "max_new_tokens": max_length,
            "attention_window_size": ATTENTION_WINDOW_SIZE
        },
        "generated_text": generated_text,
        "generated_token_ids": generated_tokens
    }
    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    # Save the hidden states and cosine similarity matrices
    for layer_name, states in hidden_states_storage.items():
        # Concatenate along the sequence dimension (dim=0) and remove the batch dim (dim=1)
        layer_states = torch.cat(states, dim=0)
        torch.save(layer_states, os.path.join(result_dir, f"hidden_states_{layer_name}.pt"))
        chunked_cosine_similarity_and_save(layer_states, result_dir, batch_size=2048, threshold=0.5, resolution=0.05, save_name=f"cosine_sim_matrix_{layer_name}.pt")
        del layer_states
        torch.cuda.empty_cache()

    print(f"Results saved to {result_dir}")

def top_p_filtering(logits, top_p=0.95, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1000 # 65536
TEMPERATURE = 0.6
ATTENTION_WINDOW_SIZE = 256 # 4096

print(f"Using device: {DEVICE}")

attn_impl = "auto"
try:
    from transformers.utils import is_flash_attn_2_available
    if is_flash_attn_2_available():
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"
except Exception:
    attn_impl = "sdpa"

# MODEL_NAME = "deepseek‑ai/DeepSeek‑R1‑Distill‑Qwen‑14B" 
# LOCAL_DIR = "deepseek_r1_14b"   # local path to save the model
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LOCAL_DIR = "."   # local path to save the model


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=LOCAL_DIR,
    # quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=LOCAL_DIR,
    trust_remote_code=True
)

print(f"Loading model: {MODEL_NAME} with attn_implementation={attn_impl}")
model.to(DEVICE)
model.eval()


prompts = [
    # Space Technology Conversation
#     """Generate a continuous conversation about interstellar propulsion systems using strict JSON format. Alternate between user questions and assistant answers. Maintain technical depth while naturally progressing through topics. Begin with:
# {"role": "user", "content": "Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. Compare chemical rockets, nuclear propulsion, laser sails, antimatter drives, and other theoretical concepts in terms of energy requirements, achievable speeds, technological feasibility, and projected timelines for development."}""",

"Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. Compare chemical rockets, nuclear propulsion, laser sails, antimatter drives, and other theoretical concepts in terms of energy requirements, achievable speeds, technological feasibility, and projected timelines for development. Include discussion of the Breakthrough Starshot initiative and other major projects. Please reason step by step, and put the final answer inside \boxed{ }."

]

prompt_names = [
    "interstellar_propulsion_review",       # Space Technology (Detailed Technical Review)
]




main_results_dir = "try_new_model" #
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