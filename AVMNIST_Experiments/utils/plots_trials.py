import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import optuna

def load_all_versions(log_dir):
    """Load all metrics.csv files from versioned experiment directories."""
    metrics_df = pd.DataFrame()
    
    for version_dir in os.listdir(log_dir):
        version_path = os.path.join(log_dir, version_dir)
        
        # Check if the directory is a "version_*" folder
        if os.path.isdir(version_path) and version_dir.startswith("version_"):
            metrics_file = os.path.join(version_path, "metrics.csv")
            if os.path.exists(metrics_file):
                version_df = pd.read_csv(metrics_file)
                version_df['version'] = version_dir  # Add version column
                metrics_df = pd.concat([metrics_df, version_df], ignore_index=True)
    
    return metrics_df

def process_metrics(df):
    """Filter out NaNs and compute the mean train_loss_step per epoch."""
    df_filtered = df.dropna(subset=["train_loss_step"])  # Remove NaN rows
    mean_train_loss = df_filtered.groupby("epoch")["train_loss_step"].mean().reset_index()
    mean_train_loss.rename(columns={"train_loss_step": "train_loss_epoch"}, inplace=True)
    return mean_train_loss

def save_versions_to_csv(df, csv_dir):
    """Save processed metrics to CSV."""
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(os.path.join(csv_dir, "metrics_versions.csv"), index=False)

def create_dir(plot_dir):
    """Ensure plot directory exists."""
    os.makedirs(plot_dir, exist_ok=True)

def plot_loss(metrics_df, plot_dir='plots/', title="Avg. Training Loss per Epoch - All Versions", filename="avg_loss_all_versions"):
    """Plot mean training loss per epoch."""
    create_dir(plot_dir)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=metrics_df, x="epoch", y="train_loss_epoch", marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(plot_dir, f"{filename}.png"))
    plt.close()


# def plot_accuracy(metrics_df, plot_dir='plots/'):
#     plt.figure(figsize=(10, 5))
#     sns.lineplot(data=metrics_df[['train_accuracy', 'val_accuracy']], dashes=False)
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.savefig(f'{plot_dir}/accuracy.png')
    
    
def plot_study_results(study, plot_dir='plots/'):
    # Extract study data
    trials = study.trials
    data = {
        "trial": [t.number for t in trials],
        "value": [t.value for t in trials],
    }
    
    # Add all hyperparameters as columns
    for param in study.best_params.keys():
        data[param] = [t.params.get(param, None) for t in trials]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Plot optimization results
    sns.lineplot(data=df, x="trial", y="value", marker='o')
    plt.title("Optimization Results")
    plt.xlabel("Trial Number")
    plt.ylabel("Objective Value")
    plt.savefig(f"{plot_dir}/optimization_results_line.png")

    # Pair plot to visualize relationships between hyperparameters and objective
    sns.pairplot(df, diag_kind="kde", corner=True)
    plt.savefig(f"{plot_dir}/optimization_results_pair.png")

# new enhanced plot code:

def load_and_preprocess_metrics(log_dir):
    """
    Load metrics and add derived columns for better analysis.
    """
    metrics_df = load_all_versions(log_dir)
    
    # Group by version and add relative progress within each version
    metrics_df['step_within_version'] = metrics_df.groupby('version')['step'].rank()
    
    # Calculate moving averages for smoothing
    metrics_df['train_loss_smooth'] = metrics_df.groupby('version')['train_loss_epoch'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Calculate change rates (derivatives)
    metrics_df['loss_change_rate'] = metrics_df.groupby('version')['train_loss_smooth'].transform(
        lambda x: x.diff() / x.shift(1)
    )
    
    return metrics_df

def plot_loss_comparison(metrics_df, plot_dir='plots/'):
    """
    Compare loss curves across different versions with confidence intervals.
    """
    plt.figure(figsize=(12, 7))
    
    # Plot each version with a different color and add confidence interval
    sns.lineplot(
        data=metrics_df, 
        x='step', 
        y='train_loss_epoch', 
        hue='version',
        errorbar=('ci', 95),
        palette='viridis'
    )
    
    plt.title('Training Loss Comparison Across Versions', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(title='Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/loss_comparison.png', dpi=300)
    plt.close()

def plot_convergence_analysis(metrics_df, plot_dir='plots/'):
    """
    Analyze convergence behavior of different model versions.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Smoothed training loss
    sns.lineplot(
        data=metrics_df, 
        x='step', 
        y='train_loss_smooth', 
        hue='version',
        ax=ax1,
        palette='viridis'
    )
    ax1.set_title('Smoothed Training Loss', fontsize=14)
    ax1.set_ylabel('Loss (5-step moving avg)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Rate of change in loss (derivative)
    sns.lineplot(
        data=metrics_df, 
        x='step', 
        y='loss_change_rate', 
        hue='version',
        ax=ax2,
        palette='viridis'
    )
    ax2.set_title('Rate of Change in Loss', fontsize=14)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Relative Change Rate', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.legend(title='Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/convergence_analysis.png', dpi=300)
    plt.close()

def plot_step_vs_loss_heatmap(metrics_df, plot_dir='plots/'):
    """
    Create a heatmap showing the distribution of loss values across steps and versions.
    """
    # Pivot the data to create a matrix of versions vs steps
    pivot_df = metrics_df.pivot_table(
        index='version', 
        columns='step', 
        values='train_loss_epoch',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(14, 8))
    
    # Create a custom colormap from blue to white
    colors = [(0, 0, 0.8), (0.5, 0.5, 1), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list('blue_to_white', colors, N=256)
    
    # Plot the heatmap
    sns.heatmap(
        pivot_df, 
        cmap=cmap,
        annot=False, 
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': 'Loss Value'}
    )
    
    plt.title('Loss Values Across Steps and Versions', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Version', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/step_vs_version_heatmap.png', dpi=300)
    plt.close()

def plot_loss_distribution(metrics_df, plot_dir='plots/'):
    """
    Plot the distribution of loss values for each version.
    """
    plt.figure(figsize=(12, 8))
    
    # Get unique versions
    versions = metrics_df['version'].unique()
    
    # Create violin plots
    sns.violinplot(
        data=metrics_df,
        x='version', 
        y='train_loss_epoch',
        hue='version',  # Assign `x` variable to `hue`
        palette='viridis',
        inner='quartile',
        cut=0,
        legend=False  # Suppress legend since hue is just a repeat of x
    )
    
    plt.title('Distribution of Loss Values by Version', fontsize=14)
    plt.xlabel('Version', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/loss_distribution.png', dpi=300)
    plt.close()

def plot_training_stability(metrics_df, plot_dir='plots/'):
    """
    Plot the stability of training across different versions.
    """
    # Calculate stability metrics for each version
    stability_df = metrics_df.groupby('version').agg({
        'train_loss_epoch': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten the multi-level columns
    stability_df.columns = ['version', 'mean_loss', 'std_loss', 'min_loss', 'max_loss']
    
    # Calculate coefficient of variation (lower is more stable)
    stability_df['variation_coef'] = stability_df['std_loss'] / stability_df['mean_loss']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Mean loss with error bars
    sns.barplot(
        data=stability_df,
        x='version',
        y='mean_loss',
        ax=axes[0],
        palette='viridis',
        hue='version',
        legend=False,
        alpha=0.7
    )
    axes[0].errorbar(
        x=range(len(stability_df)),
        y=stability_df['mean_loss'],
        yerr=stability_df['std_loss'].values,
        fmt='none',
        ecolor='black',
        capsize=5
    )

    axes[0].set_title('Mean Loss with Standard Deviation', fontsize=14)
    axes[0].set_xlabel('Version', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Coefficient of variation (stability metric)
    sns.barplot(
        data=stability_df,
        x='version',
        y='variation_coef',
        hue='version',
        legend=False,
        ax=axes[1],
        palette='viridis',
        alpha=0.7
    )
    axes[1].set_title('Training Stability (Coefficient of Variation)', fontsize=14)
    axes[1].set_xlabel('Version', fontsize=12)
    axes[1].set_ylabel('Variation Coefficient (lower is more stable)', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/training_stability.png', dpi=300)
    plt.close()

def plot_optuna_parallel_coordinates(study, plot_dir='plots/'):
    """
    Create parallel coordinates plot for hyperparameter optimization.
    """
    # Extract study data
    trials = study.trials
    data = {
        "trial": [t.number for t in trials],
        "value": [t.value for t in trials],
    }
    
    # Add all hyperparameters as columns
    for param in study.best_params.keys():
        data[param] = [t.params.get(param, None) for t in trials]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Normalize the data for parallel coordinates
    df_norm = df.copy()
    for column in df.columns:
        if column not in ['trial']:
            df_norm[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    # Create parallel coordinates plot
    plt.figure(figsize=(14, 8))
    
    # Plot each trial as a line
    for i in range(len(df)):
        # Color based on objective value (lower is better for loss)
        # Normalize value for coloring (0 to 1)
        normalized_value = (df['value'][i] - df['value'].min()) / (df['value'].max() - df['value'].min())
        # Invert so that lower loss = better color
        color_val = 1 - normalized_value
        color = plt.cm.viridis(color_val)
        
        # Get x and y coordinates for the line
        x = range(len(df.columns) - 1)  # Exclude 'trial' column
        y = df_norm.iloc[i, 1:].values  # Exclude 'trial' column
        
        plt.plot(x, y, color=color, alpha=0.6)
    
    # Add labels
    plt.xticks(range(len(df.columns) - 1), df.columns[1:], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Parallel Coordinates Plot of Hyperparameters', fontsize=14)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=df['value'].min(), vmax=df['value'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Objective Value (Loss)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/parallel_coordinates.png', dpi=300)
    plt.close()

def plot_optuna_parameter_importances(study, plot_dir='plots/'):
    """
    Visualize the importance of each hyperparameter.
    """
    # Check if we have enough trials for importance calculation
    if len(study.trials) < 5:
        print("Not enough trials to calculate parameter importances")
        return
    
    try:
        importances = optuna.importance.get_param_importances(study)
        
        # Convert to DataFrame for plotting
        importance_df = pd.DataFrame({
            'Parameter': list(importances.keys()),
            'Importance': list(importances.values())
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Parameter', palette='viridis', hue='None')
        plt.title('Hyperparameter Importance', fontsize=14)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Parameter', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/parameter_importances.png', dpi=300)
        plt.close()
    except:
        print("Could not calculate parameter importances")

def create_enhanced_plots(log_dir, study=None, plot_dir='plots/'):
    """
    Create all enhanced plots.
    """
    # Create directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load and preprocess data
    metrics_df = load_and_preprocess_metrics(log_dir)
    
    # Create all the plots
    plot_loss_comparison(metrics_df, plot_dir)
    plot_convergence_analysis(metrics_df, plot_dir)
    plot_step_vs_loss_heatmap(metrics_df, plot_dir)
    plot_loss_distribution(metrics_df, plot_dir)
    plot_training_stability(metrics_df, plot_dir)
    
    # Optuna-specific plots
    if study is not None:
        try:
            plot_optuna_parallel_coordinates(study, plot_dir)
            plot_optuna_parameter_importances(study, plot_dir)
        except ImportError:
            print("Optuna not available, skipping related plots")
        except Exception as e:
            print(f"Error creating Optuna plots: {e}")
    
    # Save processed data
    metrics_df.to_csv(f'{plot_dir}/processed_metrics.csv', index=False)
    
    print(f"All plots created and saved to {plot_dir}")


def create_plots_for_study(study=None, versions_path=None, plots_path=None):
    """
    Create plots for a given Optuna study. The end result is that a variety of 
    plots will be saved to the specified plots_path directory.
    """
    metrics_df = load_all_versions(versions_path)
    # NOTE: in order to plot a single version:
    # metrics_df = metrics_df[metrics_df["version"] == "version_0"]
    # plot_loss(metrics_df, plots_path, title="Training Loss Per Epoch - Version 0", filename="train_loss_per_epoch_version_0.png")
    processed_df = process_metrics(metrics_df)
    plot_loss(processed_df, plots_path)
    save_versions_to_csv(processed_df, plots_path)
    
    # Add the enhanced plots
    create_enhanced_plots(versions_path, study, plots_path)