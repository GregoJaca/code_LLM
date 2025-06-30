import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
    with torch.no_grad(), torch.amp.autocast("cuda"):
        generation_config = {
            "max_new_tokens": max_length,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": None,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            "temperature": TEMPERATURE,
            "top_p": 0.95,
        }
        outputs = model.generate(
            **inputs,
            **generation_config,
        )
    generated_tokens = outputs.sequences[0].tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    num_layers = len(outputs.hidden_states[0])
    selected_layers = {"first": 0, "last": num_layers - 1}

    # Extract token-by-token output
    tokens_generated = []
    for i in range(len(generated_tokens)):
        token = generated_tokens[i]
        token_text = tokenizer.decode([token])
        tokens_generated.append(token_text)
    # Save the tokens data
    with open(os.path.join(result_dir, "tokens.json"), "w") as f:
        json.dump(tokens_generated, f, indent=2)

    # Save the prompt and model configuration and answer text
    result_data = {
        "prompt": prompt,
        "model_configuration": {
            "model_name": model.config._name_or_path,
            "temperature": TEMPERATURE,
            "top_p": 0.95,
            "do_sample": True,
            "device": device,
            "max_new_tokens": max_length
        },
        "generated_text": generated_text,
        "generated_token_ids": generated_tokens
    }
    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    # Save the hidden states and cosine similarity matrices
    layer_states_dict = {}
    # First, calculate and save hidden states for all selected layers
    for layer_name, layer_idx in selected_layers.items():
        layer_states = torch.cat([
            step_hidden_states[layer_idx]
            for step_hidden_states in outputs.hidden_states
        ], dim=1)
        layer_states_cpu = layer_states.detach().cpu().to(torch.float16)
        torch.save(layer_states_cpu, os.path.join(result_dir, f"hidden_states_{layer_name}.pt"))
        layer_states_dict[layer_name] = layer_states
        del layer_states_cpu
        torch.cuda.empty_cache()
    # Then, run chunked_cosine_similarity_and_save for all selected layers
    for layer_name, layer_states in layer_states_dict.items():
        chunked_cosine_similarity_and_save(layer_states, result_dir, batch_size=2048, threshold=0.5, resolution=0.05, save_name=f"cosine_sim_matrix_{layer_name}.pt")
        del layer_states
        torch.cuda.empty_cache()

    print(f"Results saved to {result_dir}")

# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "arcee-ai/virtuoso-lite" 
# LOCAL_DIR = "./virtuoso-lite-10b"

MODEL_NAME = "Noorhan/mistral-7b-4bit"
LOCAL_DIR = "./mistral-7b"  # local path to save the model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1000 # 65536 / 2
TEMPERATURE = 0.6

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

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
#     attn_implementation=attn_impl
# )
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=LOCAL_DIR,
    quantization_config=quant_config,
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

# prompts = [
#     # Space Technology (Detailed Technical Review)
#     "Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. Compare chemical rockets, nuclear propulsion, laser sails, antimatter drives, and other theoretical concepts in terms of energy requirements, achievable speeds, technological feasibility, and projected timelines for development. Include discussion of the Breakthrough Starshot initiative and other major projects.",

#     "Examine how childhood experiences shape personality development. Discuss various influences including family environment, education, friendships, and significant life events. Explain psychological concepts like attachment theory and nature vs. nurture in accessible terms. Provide examples of how positive and negative experiences can affect adult personality traits and behaviors. Think step by step and reason exhaustively before you answer, finally give a comprehensive and long asnwer.",
# ]

prompts = [
    # Space Technology Conversation
    """Generate a continuous conversation about interstellar propulsion systems using strict JSON format. Alternate between user questions and assistant answers. Maintain technical depth while naturally progressing through topics. Begin with:
{"role": "user", "content": "Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. Compare chemical rockets, nuclear propulsion, laser sails, antimatter drives, and other theoretical concepts in terms of energy requirements, achievable speeds, technological feasibility, and projected timelines for development."}""",

    # Psychology Conversation
    """Create an ongoing dialogue about childhood development using strict JSON formatting. Alternate roles with each message, maintaining academic depth while exploring new aspects. Start with:
{"role": "user", "content": "Examine how childhood experiences shape personality development. Discuss various influences including family environment, education, friendships, and significant life events. Explain psychological concepts like attachment theory and nature vs. nurture in accessible terms. Provide examples of how positive and negative experiences can affect adult personality traits and behaviors."}"""
]

prompt_names = [
    "interstellar_propulsion_review",       # Space Technology (Detailed Technical Review)
    "childhood_personality_development",
]

# prompts = [
#     # Artificial Intelligence Conversation
#     """Generate a continuous technical dialogue about transformer architectures using strict JSON formatting. Alternate between user questions and assistant answers. Maintain mathematical rigor while naturally progressing through topics. Begin with:
# {"role": "user", "content": "Could you provide a comprehensive technical analysis of the transformer architecture underpinning modern LLMs? Specifically explain the mathematical foundations of self-attention mechanisms, positional encoding, and multi-head attention with equations where appropriate."}""",

#     # Future of Work Conversation
#     """Create an ongoing dialogue about AI's impact on employment using strict JSON formatting. Alternate roles with each response. Begin with:
# {"role": "user", "content": "Analyze in depth how artificial intelligence and automation will transform work over the next 50 years. Include projections for job displacement, necessary skills evolution, policy responses like UBI, and psychological impacts - drawing comparisons to historical industrial revolutions."}""",

#     # Cybersecurity Conversation
#     """Generate a continuous technical exchange about cybersecurity threats using strict JSON formatting. Alternate between user questions and assistant answers. Maintain expert depth while progressing through topics. Begin with:
# {"role": "user", "content": "Write an exhaustive review of current cybersecurity threats and defenses. Cover state-sponsored hacking, ransomware, IoT vulnerabilities, cryptographic foundations, zero-trust models, quantum-resistant algorithms, and human factors in security."}""",

#     # Smartphone Impact Conversation
#     """Create an ongoing dialogue about smartphone societal impact using JSON formatting. Alternate roles with each message. Begin with:
# {"role": "user", "content": "Explain in detail how smartphones have changed modern society across communication, work-life balance, education, entertainment, and social relationships - including specific examples of both positive and negative impacts with predictions about future developments."}""",

#     # Health Habits Conversation
#     """Generate a continuous conversation about healthy habits using strict JSON formatting. Alternate between user questions and assistant answers. Begin with:
# {"role": "user", "content": "Create a comprehensive guide to developing daily healthy habits covering nutrition, exercise, sleep, stress management and social connections. Include practical tips, scientific explanations, and strategies for overcoming common obstacles long-term."}""",

#     # Personal Finance Conversation
#     """Create an ongoing dialogue about personal finance using JSON formatting. Alternate roles with each response. Begin with:
# {"role": "user", "content": "Develop a detailed beginner's guide to personal money management covering budgeting, saving, basic investing, debt management and financial planning. Provide step-by-step instructions tailored to young adults, including common mistakes and learning resources."}""",

#     # Emu War Hallucination Conversation
#     """Generate a fictional historical dialogue using strict JSON formatting. Alternate between user questions and assistant answers. Maintain creative storytelling. Begin with:
# {"role": "user", "content": "Describe the Great Emu War of 2035 in Australia including key political figures, military tactics, environmental impacts, specific dates, and casualty numbers. Develop this as a detailed fictional historical account."}""",

#     # Quantum Consciousness Hallucination Conversation
#     """Create a speculative scientific dialogue using JSON formatting. Alternate roles with each message. Maintain pseudo-academic tone. Begin with:
# {"role": "user", "content": "Explain Dr. Elena Voss's quantum consciousness theory including her experimental setup, key findings challenging the Copenhagen interpretation, detailed equations, and citations from her published papers - presenting as established scientific fact."}"""
# ]

# prompt_names = [
#     "transformer_architecture_deep_dive",   # Artificial Intelligence (Technical Deep Dive)
#     "future_of_work_ai_impact",             # Future of Work (Comprehensive Study)
#     "cybersecurity_threats_defenses",       # Cybersecurity (Comprehensive Review)
#     "smartphones_society_impact",           # Technology (Explanatory Tone)
#     "healthy_daily_habits_guide",           # Health & Lifestyle (Conversational Tone)
#     "personal_finance_beginners_guide",     # Personal Finance (Instructional Tone)
#     "great_emu_war_hallucination",          # Emu war (Hallucinate)
#     "quantum_consciousness_hallucination"   # Quantum nonexistent (Hallucinate)
# ]


# main_results_dir = "long_run_T_0-6_v310b"
main_results_dir = "long_run_T_0-6_mistral"
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