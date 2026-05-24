# import torch
# import os
# import random
# from datasets import load_dataset
# from transformers import (
#     AutoProcessor,
#     Idefics3ForConditionalGeneration,
#     TrainingArguments,
#     Trainer
# )
# from peft import LoraConfig, get_peft_model

# def run_finetune():
#     print("Initializing LoRA Fine-tuning pipeline for SmolVLM...")

#     # 1. =========== Model & Processor =============
#     model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
#     processor = AutoProcessor.from_pretrained(model_id)
    
#     model = Idefics3ForConditionalGeneration.from_pretrained(
#         model_id, 
#         torch_dtype=torch.bfloat16,
#         device_map="auto" 
#     )

#     # 2. =========== Apply LoRA =============
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
#     model = get_peft_model(model, lora_config)

#     # 3. =========== Spatial Dataset Preparation =============
#     print("Loading working dataset and filtering for spatial tasks...")
#     dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train[:25000]")

#     spatial_words = ["left", "right", "behind", "front", "closest", "furthest", "next to", "count", "how many"]
    
#     def is_spatial(example):
#         try:
#             text = example["messages"][0]["content"][1]["text"]
#             if isinstance(text, str):
#                 return any(word in text.lower() for word in spatial_words)
#             return False
#         except:
#             return False 

#     dataset = dataset.filter(is_spatial)
    
#     if len(dataset) > 2000:
#         dataset = dataset.select(range(2000))
        
#     print(f"Successfully created a dataset of {len(dataset)} spatial examples!")

#     def format_data(example):
#         """Converts conversational answers into a multiple-choice format."""
#         user_text = example["messages"][0]["content"][1].get("text", "") or ""
#         true_answer = example["messages"][1]["content"][0].get("text", "") or ""
        
#         # --- MULTIPLE CHOICE GENERATOR ---
#         # Generate dummy wrong answers to force the model to learn multiple-choice formats
#         dummy_choices = [
#             "I cannot determine the spatial arrangement.",
#             "The image is too blurry to count.",
#             "The objects are arranged differently."
#         ]
        
#         choices = [true_answer] + dummy_choices
#         random.shuffle(choices)

#         choices_text = " ".join([f"({chr(97+i)}) {c}" for i, c in enumerate(choices)])
#         prompt_text = f"{user_text}\nOptions: {choices_text}\nAnswer strictly with only the correct option letter, like (a) or (b)."

#         correct_letter = chr(97 + choices.index(true_answer))
#         final_answer = f"({correct_letter})" 
#         # ---------------------------------
        
#         messages = [
#             {
#                 "role": "user",
#                 "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]
#             },
#             {
#                 "role": "assistant",
#                 "content": [{"type": "text", "text": final_answer}]
#             }
#         ]
        
#         prompt = processor.apply_chat_template(messages, tokenize=False)
#         return {"text": prompt, "image": example["images"][0]}

#     train_dataset = dataset.map(format_data, remove_columns=dataset.column_names)

#     # 4. =========== Data Collator =============
#     def collate_fn(examples):
#         texts = [e["text"] for e in examples]
#         images = [e["image"].convert("RGB") for e in examples]
        
#         batch = processor(
#             text=texts, 
#             images=images, 
#             padding=True, 
#             return_tensors="pt",
#             size={"longest_edge": 384} 
#         )

#         labels = batch["input_ids"].clone()
#         labels[labels == processor.tokenizer.pad_token_id] = -100
#         batch["labels"] = labels
        
#         return batch

#     # 5. =========== Training Loop =============
#     output_dir = "experiments/depth_finetune"
    
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         learning_rate=2e-4,
#         num_train_epochs=4,                 # BUMPED TO 4 EPOCHS
#         logging_steps=10,
#         save_strategy="epoch",
#         optim="adamw_torch",
#         remove_unused_columns=False,
#         report_to="none"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=collate_fn,
#     )

#     print("Starting Advanced Multiple-Choice Training...")
#     trainer.train()

#     # 6. =========== Save the New Weights =============
#     print(f"Training complete. Saving adapter to {output_dir}")
#     trainer.save_model(output_dir)
#     processor.save_pretrained(output_dir)

# if __name__ == "__main__":
#     run_finetune()








import sys
import os
from transformers import TrainingArguments, Trainer, set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vlm.smolvlm import get_smolvlm_model_and_processor, apply_lora_to_model
from datasets.preprocess import load_and_filter_spatial_dataset, prepare_training_dataset, get_data_collator

def run_finetune():
    print("Initializing Modular LoRA Fine-tuning pipeline...")
    
    set_seed(42) # LOCK GLOBAL RANDOMNESS

    model, processor = get_smolvlm_model_and_processor()
    model = apply_lora_to_model(model)

    raw_dataset = load_and_filter_spatial_dataset(target_samples=2000)
    train_dataset = prepare_training_dataset(raw_dataset, processor)
    collate_fn = get_data_collator(processor)

    output_dir = "experiments/depth_finetune"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=4,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
        seed=42 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    print("Starting Advanced Multiple-Choice Training...")
    trainer.train()

    print(f"Training complete. Saving adapter to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    run_finetune()


