import random
import datasets as hf_datasets

def load_and_filter_spatial_dataset(split_limit=25000, target_samples=2000):
    print("Loading working dataset and filtering for spatial tasks...")
    dataset = hf_datasets.load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split=f"train[:{split_limit}]")

    spatial_words = ["left", "right", "behind", "front", "closest", "furthest", "next to", "count", "how many"]
    
    def is_spatial(example):
        try:
            text = example["messages"][0]["content"][1]["text"]
            if isinstance(text, str):
                return any(word in text.lower() for word in spatial_words)
            return False
        except:
            return False 

    dataset = dataset.filter(is_spatial)
    
    if len(dataset) > target_samples:
        dataset = dataset.select(range(target_samples))
        
    print(f"Successfully created a dataset of {len(dataset)} spatial examples!")
    return dataset

def prepare_training_dataset(dataset, processor):
    def format_data(example):
        user_text = example["messages"][0]["content"][1].get("text", "") or ""
        true_answer = str(example["messages"][1]["content"][0].get("text", "") or "").strip()
        
        dummy_choices = [
            "Cannot determine",
            "Image is too blurry",
            "None of the above"
        ]
        
        choices = [true_answer] + dummy_choices
        random.shuffle(choices)

        choices_text = " ".join([f"({chr(97+i)}) {c}" for i, c in enumerate(choices)])
        prompt_text = f"{user_text}\nOptions: {choices_text}\nAnswer strictly with only the correct option letter, like (a) or (b)."

        correct_letter = chr(97 + choices.index(true_answer))
        final_answer = f"({correct_letter})" 
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": final_answer}]}
        ]
        
        prompt = processor.apply_chat_template(messages, tokenize=False)
        return {"text": prompt, "image": example["images"][0]}

    return dataset.map(format_data, remove_columns=dataset.column_names)

def get_data_collator(processor, max_image_edge=384):
    def collate_fn(examples):
        texts = [e["text"] for e in examples]
        images = [e["image"].convert("RGB") for e in examples]
        
        batch = processor(
            text=texts, 
            images=images, 
            padding=True, 
            return_tensors="pt",
            size={"longest_edge": max_image_edge} 
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch
    return collate_fn





