import os
import json
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from src.utils import ensure_dir, top_p_filtering
from src.config import Default

def generate_fixed_length(
    model, tokenizer, initial_embeddings, attention_mask, max_length, selected_layers, device, 
    temperature, top_p, top_k, repetition_penalty):
    n_conditions = initial_embeddings.shape[0]
    
    hidden_states_storages = {layer: [] for layer in selected_layers}
    generated_texts = []
    generated_token_ids = []

    for i in range(n_conditions):
        current_embedding = initial_embeddings[i].unsqueeze(0)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=current_embedding,
                attention_mask=attention_mask,
                num_return_sequences=1,
                max_length=max_length,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_k=top_k if ((top_k > 0) and (temperature > 0)) else None,
                top_p=top_p if ((top_p < 1.0) and (temperature > 0)) else None,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                eos_token_id=None
            )

            for layer_idx in selected_layers:
                layer_hidden_states = []
                for step_hidden_states in outputs.hidden_states:
                    state = step_hidden_states[layer_idx][0, 0, :].to(device)
                    layer_hidden_states.append(state)
                
                trajectory = torch.stack(layer_hidden_states, dim=0)
                hidden_states_storages[layer_idx].append(trajectory)

            generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            generated_token_ids.append(outputs.sequences[0].tolist())

    for layer_idx in selected_layers:
        if hidden_states_storages[layer_idx]:
            hidden_states_storages[layer_idx] = torch.stack(hidden_states_storages[layer_idx])

    return hidden_states_storages, generated_texts, generated_token_ids


def generate_text_and_hidden_states(prompt, model, tokenizer, device, max_length, result_dir, use_context_injection, context_injection_prompt, temperature, top_p, repetition_penalty, attention_window_size):
    ensure_dir(result_dir)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    num_layers = model.config.num_hidden_layers
    selected_layers = {"first": 0, "last": num_layers - 1}
    
    hidden_states_storage_cpu = {layer_name: [] for layer_name in selected_layers.keys()}
    temp_hidden_states_gpu = {layer_name: [] for layer_name in selected_layers.keys()}
    
    generated_tokens = input_ids.squeeze().tolist()
    past_key_values = None
    
    SLIDING_WINDOW_STRIDE = 64
    CONTEXT_INJECTION_INTERVAL = 2048
    total_sequence_length_for_cache = len(generated_tokens)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(max_length):
            if use_context_injection and context_injection_prompt and (len(generated_tokens) % CONTEXT_INJECTION_INTERVAL == 0) and len(generated_tokens) > 1:
                print(f"--- Injecting context at token {len(generated_tokens)}. Cache length: {past_key_values.get_seq_length()} ---")
                injection_ids = tokenizer(context_injection_prompt, return_tensors="pt").input_ids.to(device)
                injection_pos_ids = torch.arange(total_sequence_length_for_cache, total_sequence_length_for_cache + injection_ids.shape[1], device=device).unsqueeze(0)

                outputs = model(
                    input_ids=injection_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                    position_ids=injection_pos_ids
                )
                past_key_values = outputs.past_key_values
                total_sequence_length_for_cache += injection_ids.shape[1]

            if past_key_values is not None:
                current_input_ids = input_ids[:, -1:]
                position_ids = torch.tensor([[total_sequence_length_for_cache - 1]], device=device, dtype=torch.long)
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

            outputs = model(**model_inputs)
            
            next_token_logits = outputs.logits[:, -1, :]

            next_token_logits = next_token_logits / temperature
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            
            generated_tokens.append(next_token.item())
            total_sequence_length_for_cache += 1
            
            input_ids = next_token

            for layer_name, layer_idx in selected_layers.items():
                state_idx = 0 if layer_name == "first" else num_layers
                hidden_state = outputs.hidden_states[state_idx][:, -1, :].to(torch.float16)
                temp_hidden_states_gpu[layer_name].append(hidden_state)

            past_key_values = outputs.past_key_values

            if past_key_values is not None and past_key_values.get_seq_length() > attention_window_size + SLIDING_WINDOW_STRIDE:
                num_states_to_keep_on_gpu = attention_window_size
                num_states_on_gpu = len(temp_hidden_states_gpu["first"])
                num_states_to_offload = num_states_on_gpu - num_states_to_keep_on_gpu

                if num_states_to_offload > 0:
                    for layer_name in temp_hidden_states_gpu:
                        states_to_offload = temp_hidden_states_gpu[layer_name][:num_states_to_offload]
                        if states_to_offload:
                            chunk_to_offload = torch.cat(states_to_offload, dim=0).cpu()
                            hidden_states_storage_cpu[layer_name].append(chunk_to_offload)
                            temp_hidden_states_gpu[layer_name] = temp_hidden_states_gpu[layer_name][num_states_to_offload:]
                
                new_cache = []
                for key, value in past_key_values:
                    new_key = key[:, :, -attention_window_size:]
                    new_value = value[:, :, -attention_window_size:]
                    new_cache.append((new_key, new_value))
                past_key_values = DynamicCache.from_legacy_cache(past_key_values=tuple(new_cache))

    for layer_name in temp_hidden_states_gpu:
        if temp_hidden_states_gpu[layer_name]:
            remaining_gpu_chunk = torch.cat(temp_hidden_states_gpu[layer_name], dim=0).cpu()
            hidden_states_storage_cpu[layer_name].append(remaining_gpu_chunk)
            
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    tokens_generated_text = [tokenizer.decode([token]) for token in generated_tokens]
    with open(os.path.join(result_dir, "tokens.json"), "w") as f:
        json.dump(tokens_generated_text, f, indent=2)

    result_data = {
        "prompt": prompt,
        "model_configuration": {
            "model_name": model.config._name_or_path,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "repetition_penalty": repetition_penalty,
            "device": device,
            "max_new_tokens": max_length,
            "attention_window_size": attention_window_size,
            "use_context_injection": use_context_injection
        },
        "generated_text": generated_text,
        "generated_token_ids": generated_tokens
    }
    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    for layer_name, states_chunks in hidden_states_storage_cpu.items():
        if states_chunks:
            layer_states = torch.cat(states_chunks, dim=0)
            torch.save(layer_states, os.path.join(result_dir, f"hidden_states_{layer_name}.pt"))
            del layer_states
            torch.cuda.empty_cache()

    print(f"Results saved to {result_dir}")
