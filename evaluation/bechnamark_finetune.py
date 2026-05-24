import torch
from datasets import load_dataset
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from tqdm import tqdm
from peft import PeftModel


def run_baseline():
    print("Loading SmolVLM and CV-Bench dataset...")

    # 1. =========== Model =============
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    ).to("cuda") 

    adapter_path = "experiments/depth_finetune"
    model = PeftModel.from_pretrained(base_model, adapter_path)

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

        # if choices:
        #     choices_text = " ".join([f"({chr(97+i)}) {c}" for i, c in enumerate(choices)])
        #     prompt_text = f"{question}\nOptions: {choices_text}\nAnswer strictly with only the correct option letter, like (a) or (b)."
        # else:
        #     prompt_text = f"{question}\nAnswer strictly with only the correct option letter, like (a) or (b)."


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
            generated_ids = model.generate(**inputs, max_new_tokens=10,do_sample=False,temperature=None,top_p=None)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_len:]
        
        model_answer = processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

        # ===== check the gues =====
        if correct_predictions == 0 and total_samples - len(dataset) < 3:
             print(f"\nQ: {question} | True: {true_answer} | Model guessed: {model_answer}")
        # --------------------------------------

            
         # 4. ============== Answer Matching ===============
        clean_true_letter = true_answer.replace("(", "").replace(")", "").strip()
        clean_model_words = model_answer.replace(":", " ").replace("(", " ").replace(")", " ").replace(".", " ").split()

        is_correct = False
        if clean_true_letter in clean_model_words:
            is_correct = True
        else:
            try:
                true_index = ord(clean_true_letter) - 97 
                if 0 <= true_index < len(choices):
                    true_value = str(choices[true_index]).strip().lower()
                    if true_value in clean_model_words:
                        is_correct = True
            except:
                pass 

        if is_correct:
            correct_predictions += 1

    # 5. =============== Print Baseline Result ===============
    accuracy = (correct_predictions / total_samples) * 100
    print(f"\n--- Finetuned Results ---")
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_baseline()