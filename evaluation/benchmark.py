import argparse
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from tqdm import tqdm
from peft import PeftModel

def run_benchmark(adapter_path=None, longest_edge=384):
    print("Loading SmolVLM and CV-Bench dataset...")

    # 1. =========== Model Initialization =============
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    ).to("cuda") 

    model_type = "Baseline"
    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model_type = "Fine-Tuned"

    # 2. =========== Dataset =============
    dataset = load_dataset("nyu-visionx/CV-Bench", split="test")
    
    correct_predictions = 0
    total_samples = len(dataset)
    print(f"Evaluating {total_samples} samples using {model_type} model at {longest_edge}px...")
    
    # 3. ========== Run Evaluation ===============
    for item in tqdm(dataset):
        image = item["image"].convert("RGB")
        choices = item.get("choices", [])
        question = item["question"]

        # --- DYNAMIC PROMPT SWITCHING ---
        if model_type == "Fine-Tuned":
            if choices:
                choices_text = "\n".join([f"({chr(97+i)}) {c}" for i, c in enumerate(choices)])
                prompt_text = (
                    f"Question: {question}\n"
                    f"Options:\n{choices_text}\n"
                    "Instructions: You must answer by providing ONLY the single character of the correct option enclosed in parentheses. Do not provide the text of the answer. Do not explain your reasoning. For example, output '(a)' and nothing else."
                )
            else:
                prompt_text = (
                    f"Question: {question}\n"
                    "Instructions: You must answer by providing ONLY the single character of the correct option enclosed in parentheses. Do not provide the text of the answer. Do not explain your reasoning. For example, output '(a)' and nothing else."
                )
        else:
            if choices:
                choices_text = " ".join([f"({chr(97+i)}) {c}" for i, c in enumerate(choices)])
                prompt_text = f"{question}\nOptions: {choices_text}\nAnswer strictly with only the correct option letter, like (a) or (b)."
            else:
                prompt_text = f"{question}\nAnswer strictly with only the correct option letter, like (a) or (b)."

        true_answer = str(item["answer"]).strip().lower()

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = processor(text=prompt, images=[image], return_tensors="pt", size={"longest_edge": longest_edge}).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        model_answer = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip().lower()

        # 4. ============== STRICT Answer Matching ===============
        clean_true = true_answer.replace("(", "").replace(")", "").strip()
        clean_model_words = model_answer.replace(":", " ").replace("(", " ").replace(")", " ").split()

        if clean_true in clean_model_words:
            correct_predictions += 1

    # 5. =============== Print Result ===============
    accuracy = (correct_predictions / total_samples) * 100
    print(f"\n--- {model_type} Results ---")
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline or Fine-Tuned SmolVLM")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter.")
    parser.add_argument("--edge", type=int, default=384, help="Longest edge resolution for images.")
    args = parser.parse_args()
    
    run_benchmark(adapter_path=args.adapter, longest_edge=args.edge)