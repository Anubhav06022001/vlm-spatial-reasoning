import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300, 
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def plot_accuracy_comparison(output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Progression Data 
    data_progression = {
        "Experiment": ["Baseline\n(Zero-Shot)", "Exp 1\n(Standard LoRA)", "Exp 2\n(Task-Aligned)", "Exp 3\n(Expanded Vocab)"],
        "Accuracy (%)": [43.10, 42.04, 47.84, 45.11]
    }
    df_progression = pd.DataFrame(data_progression)

    # 2. Create Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Custom palette to highlight the winning Exp 2 in deep orange
    colors = ["#4C72B0", "#8FAADC", "#DD8452", "#E8B08D"]
    sns.barplot(data=df_progression, x="Experiment", y="Accuracy (%)", ax=ax, palette=colors)
    
    ax.set_title("Model Accuracy Progression Across Experiments", pad=20, fontweight='bold')
    ax.set_ylim(35, 50)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)", fontweight='bold')
    

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 8), 
                    textcoords='offset points')

    plt.tight_layout()

    save_path = os.path.join(output_dir, "spatial_accuracy_comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Professional plot successfully saved to: {save_path}")

if __name__ == "__main__":
    set_academic_style()
    plot_accuracy_comparison()