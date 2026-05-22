import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

def run_baseline():
    print("Loading SmolVLM and CV-Bench dataset...")

    # 1. =========== Model =============
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    ).to("cuda") 

    # 2. =========== Dataset =============
    dataset = load_dataset("nyu-visionx/CV-Bench", split="test")
    
    correct_predictions = 0
    total_samples = len(dataset)
    print(f"Evaluating {total_samples} samples. This will take a few minutes...")
    
    # 3. ========== Run ===============
    for item in tqdm(dataset):
        image = item["image"].convert("RGB")
        choices = item.get("choices", [])
        question = item["question"]

        if choices:
            choices_text = " ".join([f"({chr(97+i)}) {c}" for i, c in enumerate(choices)])
            prompt_text = f"{question}\nOptions: {choices_text}\nAnswer strictly with only the correct option letter, like (a) or (b)."
        else:
            prompt_text = f"{question}\nAnswer strictly with only the correct option letter, like (a) or (b)."



        true_answer = str(item["answer"]).strip().lower()

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt", size={"longest_edge": 384}).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_len:]
        
        model_answer = processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

        # ===== check the gues =====
        if correct_predictions == 0 and total_samples - len(dataset) < 3:
             print(f"\nQ: {question} | True: {true_answer} | Model guessed: {model_answer}")
        # --------------------------------------

        # 4. ============== check correct answer by matching ===============
        clean_true = true_answer.replace("(", "").replace(")", "").strip()
        clean_model_words = model_answer.replace(":", " ").replace("(", " ").replace(")", " ").split()


        if clean_true in clean_model_words:
            correct_predictions += 1

    # 5. =============== Print Baseline Result ===============
    accuracy = (correct_predictions / total_samples) * 100
    print(f"\n--- Baseline Results ---")
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_baseline()