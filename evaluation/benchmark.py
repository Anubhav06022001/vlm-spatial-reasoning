# import torch
# from datasets import load_dataset
# from transformers import AutoProcessor, Idefics3ForConditionalGeneration
# from tqdm import tqdm

# def run_baseline():
#     print("Loading SmolVLM and CV-Bench dataset...")

#     # 1. =========== Model =============
#     model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
#     processor = AutoProcessor.from_pretrained(model_id)
#     model = Idefics3ForConditionalGeneration.from_pretrained(
#         model_id, 
#         torch_dtype=torch.bfloat16
#     ).to("cuda") 

#     # 2. =========== Dataset =============
#     dataset = load_dataset("nyu-visionx/CV-Bench", split="test")
    
#     correct_predictions = 0
#     total_samples = len(dataset)
#     print(f"Evaluating {total_samples} samples. This will take a few minutes...")
    
#     # 3. ========== Run ===============
#     for item in tqdm(dataset):
#         image = item["image"].convert("RGB")
#         choices = item.get("choices", [])
#         question = item["question"]

#         # if choices:
#         #     choices_text = " ".join([f"({chr(97+i)}) {c}" for i, c in enumerate(choices)])
#         #     prompt_text = f"{question}\nOptions: {choices_text}\nAnswer strictly with only the correct option letter, like (a) or (b)."
#         # else:
#         #     prompt_text = f"{question}\nAnswer strictly with only the correct option letter, like (a) or (b)."

#         if choices:
#             choices_text = "\n".join([f"({chr(97+i)}) {c}" for i, c in enumerate(choices)])
#             prompt_text = (
#                 f"Question: {question}\n"
#                 f"Options:\n{choices_text}\n"
#                 "Instructions: You must answer by providing ONLY the single character of the correct option enclosed in parentheses. Do not provide the text of the answer. Do not explain your reasoning. For example, output '(a)' and nothing else."
#             )
#         else:
#             prompt_text = (
#                 f"Question: {question}\n"
#                 "Instructions: You must answer by providing ONLY the single character of the correct option enclosed in parentheses. Do not provide the text of the answer. Do not explain your reasoning. For example, output '(a)' and nothing else."
#             )

#         true_answer = str(item["answer"]).strip().lower()

#         messages = [
#             {
#                 "role": "user",
#                 "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]
#             }
#         ]

#         prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
#         inputs = processor(text=prompt, images=[image], return_tensors="pt", size={"longest_edge": 384}).to("cuda")

#         with torch.no_grad():
#             generated_ids = model.generate(**inputs, max_new_tokens=10)

#         input_len = inputs["input_ids"].shape[1]
#         new_tokens = generated_ids[0][input_len:]
        
#         model_answer = processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

#         # ===== check the gues =====
#         if correct_predictions == 0 and total_samples - len(dataset) < 3:
#              print(f"\nQ: {question} | True: {true_answer} | Model guessed: {model_answer}")
#         # --------------------------------------

#         # 4. ============== check correct answer by matching ===============
#         # clean_true = true_answer.replace("(", "").replace(")", "").strip()
#         # clean_model_words = model_answer.replace(":", " ").replace("(", " ").replace(")", " ").split()


#         # if clean_true in clean_model_words:
#         #     correct_predictions += 1

#          # 4. ============== Smart Answer Matching ===============
           
#        # Clean the true letter (e.g., "(c)" becomes "c")
#         clean_true_letter = true_answer.replace("(", "").replace(")", "").strip()
#         # Clean the model's output to look for letters or numbers
#         clean_model_words = model_answer.replace(":", " ").replace("(", " ").replace(")", " ").replace(".", " ").split()

#         is_correct = False
        
#         # Condition A: Did the model output the correct letter? (e.g., "c")
#         if clean_true_letter in clean_model_words:
#             is_correct = True
#         else:
#             # Condition B: Did the model output the correct actual value? (e.g., "6")
#             try:
#                 # Convert letter to index ('a'=0, 'b'=1, 'c'=2)
#                 true_index = ord(clean_true_letter) - 97 
#                 if 0 <= true_index < len(choices):
#                     true_value = str(choices[true_index]).strip().lower()
#                     # Check if the exact value text is in the model's response
#                     if true_value in clean_model_words:
#                         is_correct = True
#             except:
#                 pass # Fallback in case of weird formatting

#         if is_correct:
#             correct_predictions += 1
        
#             ##############################

#     # 5. =============== Print Baseline Result ===============
#     accuracy = (correct_predictions / total_samples) * 100
#     print(f"\n--- Baseline Results ---")
#     print(f"Final Accuracy: {accuracy:.2f}%")

# if __name__ == "__main__":
#     run_baseline()

















import argparse
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from tqdm import tqdm
from peft import PeftModel

adapter_path = "experiments/depth_finetune"

def run_benchmark(adapter_path=None, longest_edge=448):
    print("Loading SmolVLM and CV-Bench dataset...")

    # 1. =========== Model Initialization =============
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    ).to("cuda") 

    # If an adapter path is provided, load the fine-tuned LoRA weights
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

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = processor(text=prompt, images=[image], return_tensors="pt", size={"longest_edge": longest_edge}).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)

        input_len = inputs["input_ids"].shape[1]
        model_answer = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip().lower()

        # 4. ============== Smart Answer Matching ===============
        clean_true_letter = true_answer.replace("(", "").replace(")", "").strip()
        clean_model_words = model_answer.replace(":", " ").replace("(", " ").replace(")", " ").replace(".", " ").split()

        is_correct = False
        if clean_true_letter in clean_model_words:
            is_correct = True
        else:
            try:
                true_index = ord(clean_true_letter) - 97 
                if 0 <= true_index < len(choices):
                    if str(choices[true_index]).strip().lower() in clean_model_words:
                        is_correct = True
            except:
                pass 

        if is_correct:
            correct_predictions += 1

    # 5. =============== Print Result ===============
    accuracy = (correct_predictions / total_samples) * 100
    print(f"\n--- {model_type} Results ---")
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline or Fine-Tuned SmolVLM")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter. If None, evaluates baseline.")
    parser.add_argument("--edge", type=int, default=448, help="Longest edge resolution for images.")
    args = parser.parse_args()
    
    run_benchmark(adapter_path=args.adapter, longest_edge=args.edge)