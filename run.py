import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Spatial VLM Pipeline Orchestrator")
    parser.add_argument("--mode", choices=["train", "eval-base", "eval-tuned", "visualize"], required=True, 
                        help="Select which part of the pipeline to run.")
    args = parser.parse_args()

    print(f"--- Starting {args.mode.upper()} Pipeline ---")
    
    try:
        if args.mode == "train":
            subprocess.run([sys.executable, "training/finetune.py"], check=True)
        elif args.mode == "eval-base":
            subprocess.run([sys.executable, "evaluation/benchmark.py"], check=True)
        elif args.mode == "eval-tuned":
            subprocess.run([sys.executable, "evaluation/benchmark.py", "--adapter", "experiments/depth_finetune"], check=True)
        elif args.mode == "visualize":
            subprocess.run([sys.executable, "evaluation/visualize.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed during {args.mode}. Error: {e}")

if __name__ == "__main__":
    main()