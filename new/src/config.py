import torch

class Default:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    MAX_LENGTH = 4096
    REPETITION_PENALTY = 1.1
    
    # For launch_pentek.py
    N_INITIAL_CONDITIONS = 16
    RESULTS_DIR = "./launch_aug"
    SELECTED_LAYERS = [-1]

class Experiment:
    RADII = [0.04]
    TEMPS = [0.6]
    TOP_PS = [1]
    TOP_KS = [1]

class Analysis:
    SAVE_PLOTS = True
    PAIRS_TO_PLOT = [[0, 1], [0, -1], [1, 2]]
    SLIDING_WINDOW_SIZE = 16
    SLIDING_WINDOW_DISPLACEMENT = 16
    MINIMUM_VARIANCE_EXPLANATION = 0.9

# Model Configurations
class ModelConfig:
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    LOCAL_DIR = "." # "deepseek_r1_14b"

    # MODEL_NAME = "arcee-ai/virtuoso-lite" 
    # LOCAL_DIR = "./virtuoso-lite-10b" 

    # MODEL_NAME = "Noorhan/mistral-7b-4bit"
    # LOCAL_DIR = "./mistral-7b"  # local path to save the model

    # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" 
    # LOCAL_DIR = "deepseek_r1_14b"   # local path to save the model




# Prompts
class Prompts:
    prompts = [
        # "Provide a comprehensive review of propulsion systems for interstellar travel.",

        "Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. Compare chemical rockets, nuclear propulsion, laser sails, antimatter drives, and other theoretical concepts in terms of energy requirements, achievable speeds, technological feasibility, and projected timelines for development. Include discussion of major projects in the history of the field."
    ]
    
    # Example of prompts with checkpoints
    prompts_with_checkpoints = [
        """Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel, structured into distinct sections.

    Section 1: Focus on chemical rockets and nuclear propulsion. Discuss their energy requirements, achievable speeds, and current technological feasibility.
    Section 2: Transition to more advanced concepts like laser sails and antimatter drives. Detail their theoretical principles, potential speeds, and the significant technological hurdles to overcome.
    Section 3: Conclude with a discussion of other theoretical concepts and major projects like the Breakthrough Starshot initiative. Analyze their projected timelines for development and overall impact on interstellar travel.
    """
    ]
    
    context_injection_prompts = [
        "[SYSTEM REMINDER: You are writing a comprehensive technical review of interstellar propulsion systems. Ensure accuracy, depth, and cover all specified concepts and projects.]"
    ]

    prompt_names = [
        "interstellar_propulsion_review",
        "interstellar_propulsion_review_CONVERSATION",
    ]

    # Map prompt names to their content and context injection prompts
    all_prompts_data = {
        "interstellar_propulsion_review": {
            "prompt": prompts[0],
            "context_injection": context_injection_prompts[0]
        },
    }

main_results_dir = "long_run_sliding_attention_context_injection"
