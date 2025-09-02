import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

from src.config import Default, ModelConfig, Prompts, Experiment
from src.utils import PerformanceMonitor, generate_perturbed_prompts, ensure_dir
from src.generation import generate_fixed_length
from src.analysis import (
    calculate_hypervolume_and_axes
)

def analyze_llm_chaos(initial_prompt, model, tokenizer, config):
    """Main function to analyze chaotic behavior in LLM text generation."""
    print(f"Generating {config.N_INITIAL_CONDITIONS} perturbed initial conditions...")

    _, attention_mask, perturbed_embeddings = generate_perturbed_prompts(
        model, tokenizer, initial_prompt, config.N_INITIAL_CONDITIONS, config.RADIUS_INITIAL_CONDITIONS, config.DEVICE
    )

    hidden_states, generated_texts, generated_token_ids = generate_fixed_length(
        model=model,
        tokenizer=tokenizer,
        initial_embeddings=perturbed_embeddings,
        attention_mask=attention_mask,
        max_length=config.MAX_LENGTH,
        selected_layers=config.SELECTED_LAYERS,
        device=config.DEVICE,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        top_k=config.TOP_K,
        repetition_penalty=config.REPETITION_PENALTY,
    )

    print("\nAnalyzing trajectories...")
    
    trajectories = hidden_states[-1]
    hypervolumes, axis_lengths = calculate_hypervolume_and_axes(trajectories, n_axes=4)

    if config.SAVE_RESULTS:
        ensure_dir(config.RESULTS_DIR)
        
        generation_results = []
        for i in range(config.N_INITIAL_CONDITIONS):
            tokens_generated_text = [tokenizer.decode([token]) for token in generated_token_ids[i]]
            result_data = {
                "prompt": initial_prompt,
                "model_configuration": {
                    "model_name": ModelConfig.MODEL_NAME,
                    "temperature": config.TEMPERATURE,
                    "top_p": config.TOP_P,
                    "top_k": config.TOP_K,
                    "repetition_penalty": config.REPETITION_PENALTY,
                    "radius_initial_conditions": config.RADIUS_INITIAL_CONDITIONS,
                    "device": config.DEVICE,
                    "max_new_tokens": config.MAX_LENGTH,
                },
                "generated_text": generated_texts[i],
                "generated_token_ids": generated_token_ids[i],
                "generated_tokens": tokens_generated_text
            }
            generation_results.append(result_data)

        with open(os.path.join(config.RESULTS_DIR, "results.json"), "w") as f:
            json.dump(generation_results, f, indent=2)

        for layer_idx, layer_hidden_states in hidden_states.items():
            torch.save(layer_hidden_states, os.path.join(config.RESULTS_DIR, f"hidden_states_layer_{layer_idx}.pt"))

        torch.save(hypervolumes, os.path.join(config.RESULTS_DIR, "hypervolume.pt"))
        torch.save(axis_lengths, os.path.join(config.RESULTS_DIR, "axis_lengths.pt"))

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.start()

    # model = AutoModelForCausalLM.from_pretrained(ModelConfig.MODEL_NAME, torch_dtype=Default.DTYPE, cache_dir=ModelConfig.LOCAL_DIR)
    model = AutoModelForCausalLM.from_pretrained(ModelConfig.MODEL_NAME, cache_dir=ModelConfig.LOCAL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME, cache_dir=ModelConfig.LOCAL_DIR)

    print(f"Loading model: {ModelConfig.MODEL_NAME}")
    model.to(Default.DEVICE)
    model.eval()

    for rrr in Experiment.RADII:
        for ttt in range(len(Experiment.TEMPS)):
            config = Default()
            config.TEMPERATURE = Experiment.TEMPS[ttt]
            config.TOP_P = Experiment.TOP_PS[ttt]
            config.TOP_K = Experiment.TOP_KS[ttt]
            config.RADIUS_INITIAL_CONDITIONS = rrr
            config.RESULTS_DIR = f"{Default.RESULTS_DIR}/run_{config.TEMPERATURE}_{rrr}/"
            config.SAVE_RESULTS = True

            print(" ---------------------------------- ")
            print("radius: ", config.RADIUS_INITIAL_CONDITIONS)
            print("T = ", config.TEMPERATURE)

            ensure_dir(config.RESULTS_DIR)
            analyze_llm_chaos(Prompts.prompts[0], model, tokenizer, config)

            print("\n=== Run Finished ===")

    monitor.end()
