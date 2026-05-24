import sys
import os
from transformers import TrainingArguments, Trainer, set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vlm.smolvlm import get_smolvlm_model_and_processor, apply_lora_to_model
from datasets.preprocess import load_and_filter_spatial_dataset, prepare_training_dataset, get_data_collator

def run_finetune():
    print("Initializing Modular LoRA Fine-tuning pipeline...")
    
    set_seed(42) 

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


