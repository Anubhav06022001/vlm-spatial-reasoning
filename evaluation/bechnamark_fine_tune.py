import torch
from datasets import load_dataset
# from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from tqdm import tqdm

# for testing it again fine tuned model
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
            generated_ids = model.generate(**inputs, max_new_tokens=10,do_sample=False,temperature=None,top_p=None)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_len:]
        
        model_answer = processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

        # ===== check the gues =====
        if correct_predictions == 0 and total_samples - len(dataset) < 3:
             print(f"\nQ: {question} | True: {true_answer} | Model guessed: {model_answer}")
        # --------------------------------------

        '''  
       # 4. ============== check correct answer by matching ===============
        clean_true = true_answer.replace("(", "").replace(")", "").strip()
        clean_model_words = model_answer.replace(":", " ").replace("(", " ").replace(")", " ").split()


        if clean_true in clean_model_words:
            correct_predictions += 1'''
        
        ####################################################################
            
         # 4. ============== Smart Answer Matching ===============
        # Clean the true letter (e.g., "(c)" becomes "c")
        clean_true_letter = true_answer.replace("(", "").replace(")", "").strip()
        # Clean the model's output to look for letters or numbers
        clean_model_words = model_answer.replace(":", " ").replace("(", " ").replace(")", " ").replace(".", " ").split()

        is_correct = False
        
        # Condition A: Did the model output the correct letter? (e.g., "c")
        if clean_true_letter in clean_model_words:
            is_correct = True
        else:
            # Condition B: Did the model output the correct actual value? (e.g., "6")
            try:
                # Convert letter to index ('a'=0, 'b'=1, 'c'=2)
                true_index = ord(clean_true_letter) - 97 
                if 0 <= true_index < len(choices):
                    true_value = str(choices[true_index]).strip().lower()
                    # Check if the exact value text is in the model's response
                    if true_value in clean_model_words:
                        is_correct = True
            except:
                pass # Fallback in case of weird formatting

        if is_correct:
            correct_predictions += 1
        
        ##############################################################################

    # 5. =============== Print Baseline Result ===============
    accuracy = (correct_predictions / total_samples) * 100
    print(f"\n--- Baseline Results ---")
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_baseline()