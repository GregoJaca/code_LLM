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
            "max_length": max_length,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "output_hidden_states": True,
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
        "first": 0,
        "middle": num_layers // 2,
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
                
                # Calculate L2 distance efficiently using cdist
                l2_dist = torch.cdist(tensor1_flat, tensor2_flat, p=2)
                
                # Store results
                matrix_name_cos = f"cosine_sim_{layer1_name}_{layer2_name}"
                matrix_name_l2 = f"l2_dist_{layer1_name}_{layer2_name}"

                # For similarity matrices:
                torch.save(
                    cosine_sim.detach().cpu(),
                    os.path.join(result_dir, f"{matrix_name_cos}.pt")
                )

                torch.save(
                    l2_dist.detach().cpu(),
                    os.path.join(result_dir, f"{matrix_name_l2}.pt")
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
            "max_length": max_length,
            "temperature": 0.0,
            "do_sample": False,
            "device": device
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
MAX_LENGTH = 2048*2

print(f"Using device: {DEVICE}")


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32 if DEVICE == "cuda" else torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model: {MODEL_NAME}")
# Move model to device and set eval mode (ONCE)
model.to(DEVICE)
model.eval()  # Disables dropout/batch norm if any

# List of prompts to process
prompts = [ 
    # Artificial Intelligence (Technical Deep Dive)
    "Provide a comprehensive technical analysis of the transformer architecture that underpins modern large language models. Explain in detail the mathematical foundations of self-attention mechanisms, positional encoding, and multi-head attention. Compare different variants of transformers and discuss their relative advantages in terms of computational efficiency and model performance. Include practical considerations for training and deploying transformer-based models.",

    # Future of Work (Comprehensive Study)
    "Analyze in depth how artificial intelligence and automation will transform the nature of work across different sectors over the next 50 years. Include projections for job displacement and creation, necessary skills evolution, potential policy responses (like universal basic income), and the psychological and societal impacts of these changes. Draw comparisons to previous industrial revolutions.",

    # Space Technology (Detailed Technical Review)
    "Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. Compare chemical rockets, nuclear propulsion, laser sails, antimatter drives, and other theoretical concepts in terms of energy requirements, achievable speeds, technological feasibility, and projected timelines for development. Include discussion of the Breakthrough Starshot initiative and other major projects.",

    # Cybersecurity (Comprehensive Review)
    "Write an exhaustive review of current cybersecurity threats and defense strategies. Cover state-sponsored hacking, ransomware, IoT vulnerabilities, and AI-powered attacks. Explain cryptographic foundations, network security architectures, zero-trust models, and emerging quantum-resistant algorithms. Discuss the human factors in security and the challenges of creating secure systems.",

    # Technology (Explanatory Tone)
    "Explain in detail how smartphones have changed modern society. Cover aspects like communication habits, work-life balance, education, entertainment, and social relationships. Provide specific examples for each area and discuss both positive and negative impacts. Conclude with predictions about future developments in smartphone technology and usage.",

    # Health & Lifestyle (Conversational Tone)
    "Write a comprehensive guide to developing and maintaining healthy daily habits. Cover nutrition, exercise, sleep, stress management, and social connections. Provide practical tips for each area, explain the science behind why they work, and suggest ways to overcome common obstacles to maintaining these habits long-term.",

    # Personal Finance (Instructional Tone)
    "Create a detailed beginner's guide to personal money management. Explain budgeting, saving, basic investing, debt management, and financial planning in simple terms. Provide step-by-step instructions for getting started with each aspect, common mistakes to avoid, and resources for further learning. Tailor the advice to young adults just starting their financial journey.",

    # Psychology (Exploratory Tone)
    "Examine how childhood experiences shape personality development. Discuss various influences including family environment, education, friendships, and significant life events. Explain psychological concepts like attachment theory and nature vs. nurture in accessible terms. Provide examples of how positive and negative experiences can affect adult personality traits and behaviors.",

    # Emu war (Hallucinate)
    "Tell me about the historical event known as The Great Emu War of 2035 in Australia, including the key political figures involved, the military tactics used, and the long-term environmental impacts of this conflict. Provide specific dates and casualty numbers.",

    # Quantum nonexistent (Hallucinate)
    "Explain the scientific principles behind the quantum consciousness theory proposed by Dr. Elena Voss, including her experimental setup, key findings, and how her work challenges the traditional Copenhagen interpretation of quantum mechanics. Provide detailed equations and citations from her published papers."
]

prompt_names = [
    "transformer_architecture_deep_dive",   # Artificial Intelligence (Technical Deep Dive)
    "future_of_work_ai_impact",             # Future of Work (Comprehensive Study)
    "interstellar_propulsion_review",       # Space Technology (Detailed Technical Review)
    "cybersecurity_threats_defenses",       # Cybersecurity (Comprehensive Review)
    "smartphones_society_impact",           # Technology (Explanatory Tone)
    "healthy_daily_habits_guide",           # Health & Lifestyle (Conversational Tone)
    "personal_finance_beginners_guide",     # Personal Finance (Instructional Tone)
    "childhood_personality_development",    # Psychology (Exploratory Tone)
    "great_emu_war_hallucination",          # Emu war (Hallucinate)
    "quantum_consciousness_hallucination"   # Quantum nonexistent (Hallucinate)
]

# Create main results directory
main_results_dir = "results_pecs_temperature_0-6"
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













