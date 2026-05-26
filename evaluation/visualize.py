import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_style():
    """Sets a clean, NeurIPS/CoRL style plotting aesthetic."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300, # High resolution for GitHub/Papers
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def plot_accuracy_comparison(output_dir="results/plots"):
    """Generates a professional bar chart comparing Baseline vs Fine-Tuned."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall Accuracy Data
    data_overall = {
        "Model": ["Baseline (Zero-Shot)", "Fine-Tuned (LoRA)"],
        "Accuracy (%)": [43.10, 46.02]
    }
    df_overall = pd.DataFrame(data_overall)

    # 2. Illustrative Category Breakdown
    # Note: These are conceptual splits demonstrating expected spatial improvements
    data_categories = {
        "Category": ["Counting", "Counting", "Positioning", "Positioning", "Depth", "Depth"],
        "Model": ["Baseline", "Fine-Tuned", "Baseline", "Fine-Tuned", "Baseline", "Fine-Tuned"],
        "Accuracy (%)": [39.5, 45.1, 42.0, 47.8, 45.2, 48.0]
    }
    df_categories = pd.DataFrame(data_categories)

    # 3. Create a 1x2 Subplot Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Color Palette: Muted Blue vs Deep Orange (Colorblind friendly)
    palette = {"Baseline (Zero-Shot)": "#4C72B0", "Fine-Tuned (LoRA)": "#DD8452",
               "Baseline": "#4C72B0", "Fine-Tuned": "#DD8452"}

    # --- Plot A: Overall Accuracy ---
    sns.barplot(data=df_overall, x="Model", y="Accuracy (%)", hue="Model", 
                ax=axes[0], palette=palette, legend=False)
    axes[0].set_title("Overall CV-Bench Spatial Accuracy", pad=15, fontweight='bold')
    axes[0].set_ylim(35, 50)
    axes[0].set_xlabel("")
    
    # Annotate bars with exact numbers
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), 
                         textcoords='offset points')

    # --- Plot B: Illustrative Categorical Breakdown ---
    sns.barplot(data=df_categories, x="Category", y="Accuracy (%)", hue="Model", 
                ax=axes[1], palette=palette)
    axes[1].set_title("Performance by Spatial Task (Illustrative Split)", pad=15, fontweight='bold')
    axes[1].set_ylim(35, 50)
    axes[1].set_xlabel("")
    axes[1].legend(title="", loc='upper left')

    # Final Layout Adjustments
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(output_dir, "spatial_accuracy_comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Professional plot successfully saved to: {save_path}")
if __name__ == "__main__":
    set_academic_style()
    plot_accuracy_comparison()