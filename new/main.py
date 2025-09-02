import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Default, ModelConfig, Prompts, main_results_dir
from src.utils import ensure_dir, PerformanceMonitor
from src.model_loader import load_model_and_tokenizer
from src.generation import generate_text_and_hidden_states

def main():
    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start()

    # Load model and tokenizer
    print(f"Using device: {Default.DEVICE}")
    model, tokenizer = load_model_and_tokenizer(ModelConfig.MODEL_NAME, ModelConfig.LOCAL_DIR, Default.DEVICE)

    # Ensure the main results directory exists
    ensure_dir(main_results_dir)

    # Process each prompt
    for prompt_name_key in Prompts.prompt_names:
        prompt_data = Prompts.all_prompts_data[prompt_name_key]
        prompt_to_use = prompt_data["prompt"]
        context_injection_prompt_to_use = prompt_data["context_injection"]

        prompt_dir = os.path.join(main_results_dir, prompt_name_key)
        
        print(f"\n--- Starting processing for: {prompt_name_key} ---")
        
        generate_text_and_hidden_states(
            prompt=prompt_to_use,
            model=model,
            tokenizer=tokenizer,
            device=Default.DEVICE,
            max_length=Default.MAX_LENGTH,
            result_dir=prompt_dir,
            use_context_injection=True,  # As in the original script
            context_injection_prompt=context_injection_prompt_to_use,
            temperature=Default.TEMPERATURE,
            top_p=0.95,  # As in the original script
            repetition_penalty=Default.REPETITION_PENALTY,
            attention_window_size=Default.ATTENTION_WINDOW_SIZE
        )

    print("\nAll prompts processed successfully!")

    # End performance monitoring and print stats
    monitor.end()

if __name__ == "__main__":
    main()
