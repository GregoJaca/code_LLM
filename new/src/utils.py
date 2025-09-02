import os
import time
import psutil
import torch
import torch.nn.functional as F
import json
import subprocess

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def top_p_filtering(logits, top_p=0.95, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits

def generate_perturbed_prompts(model, tokenizer, initial_prompt, n_conditions, radius, device):
    encoded = tokenizer(initial_prompt, return_tensors="pt")
    initial_tokens = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.no_grad():
        initial_embedding = model.get_input_embeddings()(initial_tokens)
    
    perturbations = torch.randn(
        n_conditions, 
        *initial_embedding.shape[1:], 
        device=device
    ) * radius
    
    perturbed_embeddings = initial_embedding + perturbations
    
    return initial_tokens, attention_mask, perturbed_embeddings

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.cpu_start = None
        self.ram_start = None

    def start(self):
        self.start_time = time.time()
        self.cpu_start = psutil.cpu_percent()
        self.ram_start = psutil.virtual_memory().used

    def end(self):
        if self.start_time is None:
            print("Performance monitor was not started.")
            return

        end_time = time.time()
        execution_time = end_time - self.start_time
        print("\n\n=== PERFORMANCE METRICS ===\n")
        print(f"Total execution time: {execution_time:.2f} seconds\n")
        print(f"CPU usage: {psutil.cpu_percent()}%\n")
        print(f"RAM used: {(psutil.virtual_memory().used - self.ram_start) / 1024 / 1024:.2f} MB\n")
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB\n")

def get_last_commit_hash():
    """Retrieves the last git commit hash."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_hash = "N/A"
    return commit_hash

def save_metadata(results_dir, args, config_classes):
    """
    Saves metadata about the run, including arguments and configuration, to a JSON file.
    """
    ensure_dir(results_dir)
    
    args_dict = vars(args)
    
    config_dict = {}
    for config_class in config_classes:
        config_dict[config_class.__name__] = {
            attr: getattr(config_class, attr)
            for attr in dir(config_class)
            if not callable(getattr(config_class, attr)) and not attr.startswith("__")
        }
    
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit_hash": get_last_commit_hash(),
        "arguments": args_dict,
        "configs": config_dict
    }
    
    metadata_path = os.path.join(results_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to {metadata_path}")