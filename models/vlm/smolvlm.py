import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import LoraConfig, get_peft_model


def get_smolvlm_model_and_processor(model_id="HuggingFaceTB/SmolVLM-500M-Instruct"):
    print(f"Loading processor and model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    return model, processor

def apply_lora_to_model(model):
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model