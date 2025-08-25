import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.hooks import RemovableHandle
from torchvision.utils import make_grid
import pandas as pd
from sklearn.manifold import TSNE
from pathlib import Path
import os

class ModelDebugger:
    def __init__(self, model, output_dir="debug_outputs"):
        """
        Initialize the model debugger
        
        Args:
            model: The DINO model to debug
            output_dir: Directory to save debug outputs
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for activations and gradients
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
        # For storing history across epochs
        self.loss_history = []
        self.grad_norm_history = {}
        self.teacher_student_similarity = []
    
    def register_hooks(self):
        """Register hooks for all modules to capture activations and gradients"""
        # Clear previous hooks
        self.remove_hooks()
        
        # Helper function to get module name
        def get_module_name(module, parent_name=""):
            for name, child in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                if list(child.children()):
                    get_module_name(child, full_name)
                else:
                    # Register forward hook for activations
                    handle = child.register_forward_hook(
                        lambda m, inp, out, name=full_name: self._store_activation(name, out)
                    )
                    self.hooks.append(handle)
                    
                    # Register backward hook for gradients
                    if hasattr(child, 'weight') and child.weight is not None:
                        handle = child.register_full_backward_hook(
                            lambda m, grad_in, grad_out, name=full_name: self._store_gradient(name, grad_out[0])
                        )
                        self.hooks.append(handle)
        
        # Register hooks for student and teacher
        get_module_name(self.model.student, "student")
        get_module_name(self.model.teacher, "teacher")
        get_module_name(self.model.student_projection, "student_projection")
        get_module_name(self.model.teacher_projection, "teacher_projection")
        
        print(f"Registered {len(self.hooks)} hooks for activation and gradient tracking")
    
    def _store_activation(self, name, output):
        """Store activation values from forward pass"""
        # Handle different output types
        if isinstance(output, tuple):
            output = output[0]
        self.activations[name] = output.detach().cpu()
    
    def _store_gradient(self, name, grad):
        """Store gradient values from backward pass"""
        if grad is not None:
            self.gradients[name] = grad.detach().cpu()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("Removed all hooks")
    
    def log_epoch_metrics(self, epoch, loss):
        """Log metrics for the current epoch"""
        self.loss_history.append((epoch, loss))
        
        # Calculate and store gradient norms
        for name, grad in self.gradients.items():
            if name not in self.grad_norm_history:
                self.grad_norm_history[name] = []
            norm = grad.norm().item()
            self.grad_norm_history[name].append((epoch, norm))
        
        # Calculate student-teacher similarity
        if 'student_projection.2' in self.activations and 'teacher_projection.2' in self.activations:
            student_out = self.activations['student_projection.2']
            teacher_out = self.activations['teacher_projection.2']
            
            # If these are batched, take mean across batch dimension
            if len(student_out.shape) > 1:
                student_out = student_out.mean(0)
                teacher_out = teacher_out.mean(0)
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                student_out.flatten().unsqueeze(0),
                teacher_out.flatten().unsqueeze(0)
            ).item()
            
            self.teacher_student_similarity.append((epoch, similarity))
    
    def plot_loss_curve(self):
        """Plot the loss curve over epochs"""
        epochs, losses = zip(*self.loss_history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, 'b-', marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        output_path = self.output_dir / "loss_curve.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Loss curve saved to {output_path}")
        
        return output_path
    
    def plot_gradient_norms(self, top_n=10):
        """Plot gradient norms for top N modules with largest gradients"""
        plt.figure(figsize=(15, 8))
        
        # Calculate average gradient norm for each module
        avg_norms = {}
        for name, history in self.grad_norm_history.items():
            _, norms = zip(*history)
            avg_norms[name] = np.mean(norms)
        
        # Get top N modules with largest gradients
        top_modules = sorted(avg_norms.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Plot gradient norms over epochs for top modules
        for name, _ in top_modules:
            epochs, norms = zip(*self.grad_norm_history[name])
            plt.plot(epochs, norms, marker='o', label=name)
        
        plt.title(f'Gradient Norms for Top {top_n} Modules')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        output_path = self.output_dir / "gradient_norms.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Gradient norms plot saved to {output_path}")
        
        return output_path
    
    def plot_student_teacher_similarity(self):
        """Plot student-teacher similarity over epochs"""
        if not self.teacher_student_similarity:
            print("No student-teacher similarity data available")
            return None
        
        epochs, similarities = zip(*self.teacher_student_similarity)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, similarities, 'g-', marker='o')
        plt.title('Student-Teacher Feature Similarity Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.grid(True)
        
        output_path = self.output_dir / "student_teacher_similarity.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Student-teacher similarity plot saved to {output_path}")
        
        return output_path
    
    def visualize_feature_maps(self, layer_name, max_channels=16):
        """Visualize feature maps for a specific layer"""
        if layer_name not in self.activations:
            print(f"Layer {layer_name} not found in activations")
            return None
        
        activations = self.activations[layer_name]
        
        # Handle different activation shapes
        if len(activations.shape) == 4:  # Conv layers: [batch, channels, height, width]
            batch_size, channels, height, width = activations.shape
            
            # If too many channels, select a subset
            if channels > max_channels:
                channel_indices = np.linspace(0, channels-1, max_channels, dtype=int)
                activations = activations[:, channel_indices]
                channels = max_channels
            
            # Create a grid of feature maps
            fig, axes = plt.subplots(min(batch_size, 4), channels, figsize=(channels*2, min(batch_size, 4)*2))
            
            if batch_size == 1 and channels == 1:
                axes = np.array([[axes]])
            elif batch_size == 1:
                axes = axes.reshape(1, -1)
            elif channels == 1:
                axes = axes.reshape(-1, 1)
            
            for b in range(min(batch_size, 4)):
                for c in range(channels):
                    axes[b, c].imshow(activations[b, c].numpy(), cmap='viridis')
                    axes[b, c].axis('off')
                    if b == 0:
                        axes[b, c].set_title(f'Ch {c}')
        
        elif len(activations.shape) == 2:  # Linear layers: [batch, features]
            batch_size, features = activations.shape
            
            # Plot as heatmap
            plt.figure(figsize=(10, 6))
            sample_idx = 0  # Just show first sample in batch
            plt.imshow(activations[sample_idx:sample_idx+1].numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'Feature Map for {layer_name} (Sample {sample_idx})')
            plt.xlabel('Feature Index')
            
        else:
            print(f"Unsupported activation shape: {activations.shape}")
            return None
        
        output_path = self.output_dir / f"feature_map_{layer_name.replace('.', '_')}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Feature map visualization saved to {output_path}")
        
        return output_path
    
    def visualize_gradient_flow(self):
        """Visualize gradient flow through the network"""
        plt.figure(figsize=(15, 10))
        
        # Get average gradient norm for each module
        avg_grads = []
        layer_names = []
        
        for name, history in self.grad_norm_history.items():
            _, norms = zip(*history)
            avg_grads.append(np.mean(norms))
            layer_names.append(name)
        
        # Sort by the order they appear in the model
        sorted_indices = range(len(layer_names))
        avg_grads = [avg_grads[i] for i in sorted_indices]
        layer_names = [layer_names[i] for i in sorted_indices]
        
        # Plot bars
        plt.bar(range(len(avg_grads)), avg_grads)
        plt.xticks(range(len(avg_grads)), layer_names, rotation=90)
        plt.xlabel('Layers')
        plt.ylabel('Average Gradient Norm')
        plt.title('Gradient Flow')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        output_path = self.output_dir / "gradient_flow.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Gradient flow visualization saved to {output_path}")
        
        return output_path
    
    def visualize_embeddings_tsne(self, teacher=True, student=True):
        """
        Visualize embeddings using t-SNE
        
        Args:
            teacher: Whether to include teacher embeddings
            student: Whether to include student embeddings
        """
        embeddings = []
        labels = []
        
        # Get student embeddings
        if student and 'student_projection.2' in self.activations:
            student_emb = self.activations['student_projection.2'].numpy()
            embeddings.append(student_emb)
            labels.extend(['student'] * student_emb.shape[0])
        
        # Get teacher embeddings
        if teacher and 'teacher_projection.2' in self.activations:
            teacher_emb = self.activations['teacher_projection.2'].numpy()
            embeddings.append(teacher_emb)
            labels.extend(['teacher'] * teacher_emb.shape[0])
        
        if not embeddings:
            print("No embeddings found for visualization")
            return None
        
        # Combine embeddings
        combined_emb = np.vstack(embeddings)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(combined_emb)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        student_mask = np.array(labels) == 'student'
        teacher_mask = np.array(labels) == 'teacher'
        
        if student and np.any(student_mask):
            plt.scatter(embeddings_2d[student_mask, 0], embeddings_2d[student_mask, 1], 
                      c='blue', label='Student', alpha=0.7)
        
        if teacher and np.any(teacher_mask):
            plt.scatter(embeddings_2d[teacher_mask, 0], embeddings_2d[teacher_mask, 1], 
                      c='red', label='Teacher', alpha=0.7)
        
        plt.legend()
        plt.title('t-SNE Visualization of Embeddings')
        
        output_path = self.output_dir / "embeddings_tsne.png"
        plt.savefig(output_path)
        plt.close()
        print(f"t-SNE visualization saved to {output_path}")
        
        return output_path
    
    def analyze_weight_distributions(self):
        """Analyze the distribution of weights in the model"""
        student_weights = {}
        teacher_weights = {}
        
        # Collect weights from student and teacher
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.startswith('student'):
                    student_weights[name] = param.detach().cpu().flatten().numpy()
                elif name.startswith('teacher'):
                    teacher_weights[name] = param.detach().cpu().flatten().numpy()
        
        # Plot histograms
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Student weights
        all_student_weights = np.concatenate(list(student_weights.values()))
        axes[0].hist(all_student_weights, bins=50, alpha=0.7)
        axes[0].set_title('Student Weights Distribution')
        axes[0].set_xlabel('Weight Value')
        axes[0].set_ylabel('Frequency')
        
        # Teacher weights
        all_teacher_weights = np.concatenate(list(teacher_weights.values()))
        axes[1].hist(all_teacher_weights, bins=50, alpha=0.7)
        axes[1].set_title('Teacher Weights Distribution')
        axes[1].set_xlabel('Weight Value')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "weight_distributions.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Weight distributions saved to {output_path}")
        
        return output_path
    
    def _check_dead_neurons(self, threshold=1e-6):
        """Check for dead neurons (neurons that always output zero)"""
        dead_neurons = {}
        
        for name, activation in self.activations.items():
            if len(activation.shape) >= 2:  # Skip 1D activations
                # Check if activation is consistently near zero
                activation_mean = activation.abs().mean(dim=0)
                dead_mask = activation_mean < threshold
                
                if torch.any(dead_mask):
                    dead_count = dead_mask.sum().item()
                    total_count = dead_mask.numel()
                    dead_neurons[name] = (dead_count, total_count, dead_count / total_count)
        
        return dead_neurons
    
    def track_center_updates(self):
        center = self.model.center.detach().cpu()
        plt.figure(figsize=(10, 6))
        plt.plot(center.numpy().flatten())
        plt.title('Center Values')
        plt.grid(True)
        output_path = self.output_dir / "center_values.png"
        plt.savefig(output_path)
        plt.close()
        return output_path

    def report_dead_neurons(self):
        """Generate a report of dead neurons"""
        dead_neurons = self._check_dead_neurons()
        
        if not dead_neurons:
            print("No dead neurons detected")
            return None
        
        # Create a DataFrame for the report
        dead_data = []
        for name, (dead_count, total_count, percentage) in dead_neurons.items():
            dead_data.append({
                'Layer': name,
                'Dead Neurons': dead_count,
                'Total Neurons': total_count,
                'Percentage': percentage * 100
            })
        
        df = pd.DataFrame(dead_data)
        df = df.sort_values('Percentage', ascending=False)
        
        # Plot as a bar chart
        plt.figure(figsize=(12, 6))
        layers = df['Layer'].values
        percentages = df['Percentage'].values
        
        plt.bar(range(len(layers)), percentages)
        plt.xticks(range(len(layers)), layers, rotation=90)
        plt.xlabel('Layer')
        plt.ylabel('Dead Neurons (%)')
        plt.title('Percentage of Dead Neurons by Layer')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        output_path = self.output_dir / "dead_neurons.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Dead neurons report saved to {output_path}")
        
        # Also save as CSV
        csv_path = self.output_dir / "dead_neurons.csv"
        df.to_csv(csv_path, index=False)
        
        return output_path
    
    def analyze_batch_statistics(self):
        """Analyze batch statistics to detect issues like internal covariate shift"""
        stats = {}
        
        for name, activation in self.activations.items():
            if len(activation.shape) >= 2:  # Skip 1D activations
                # Calculate mean and variance across batch dimension
                if len(activation.shape) == 4:  # Conv layer
                    # Reshape to [batch, channels*height*width]
                    batch_size = activation.shape[0]
                    flattened = activation.reshape(batch_size, -1)
                    mean = flattened.mean(dim=1)
                    var = flattened.var(dim=1)
                else:  # Linear layer
                    mean = activation.mean(dim=1)
                    var = activation.var(dim=1)
                
                stats[name] = {
                    'mean': mean.mean().item(),
                    'mean_std': mean.std().item(),
                    'var': var.mean().item(),
                    'var_std': var.std().item()
                }
        
        # Create a DataFrame for the report
        df = pd.DataFrame([
            {
                'Layer': name,
                'Mean': stat['mean'],
                'Mean Std': stat['mean_std'],
                'Variance': stat['var'],
                'Variance Std': stat['var_std']
            }
            for name, stat in stats.items()
        ])
        
        # Plot as bar charts
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        layers = df['Layer'].values
        x = np.arange(len(layers))
        
        # Mean plot
        axes[0].bar(x - 0.2, df['Mean'], width=0.4, label='Mean')
        axes[0].bar(x + 0.2, df['Mean Std'], width=0.4, label='Mean Std')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(layers, rotation=90)
        axes[0].set_title('Mean and Standard Deviation of Mean')
        axes[0].legend()
        axes[0].grid(True, axis='y')
        
        # Variance plot
        axes[1].bar(x - 0.2, df['Variance'], width=0.4, label='Variance')
        axes[1].bar(x + 0.2, df['Variance Std'], width=0.4, label='Variance Std')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(layers, rotation=90)
        axes[1].set_title('Mean and Standard Deviation of Variance')
        axes[1].legend()
        axes[1].grid(True, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "batch_statistics.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Batch statistics saved to {output_path}")
        
        # Also save as CSV
        csv_path = self.output_dir / "batch_statistics.csv"
        df.to_csv(csv_path, index=False)
        
        return output_path
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive debug report"""
        print("Generating comprehensive debug report...")
        
        # Create all plots
        loss_plot = self.plot_loss_curve()
        grad_norm_plot = self.plot_gradient_norms()
        similarity_plot = self.plot_student_teacher_similarity()
        grad_flow_plot = self.visualize_gradient_flow()
        tsne_plot = self.visualize_embeddings_tsne()
        weight_dist_plot = self.analyze_weight_distributions()
        dead_neurons_plot = self.report_dead_neurons()
        batch_stats_plot = self.analyze_batch_statistics()
        center_updates_plot = self.track_center_updates()
        
        # Feature maps for key layers
        feature_map_plots = []
        key_layers = [
            'student.encoder.0',  # First conv layer
            'student.encoder.4',  # Middle conv layer
            'student.encoder.8',  # Last conv layer
            'student_projection.0',  # First projection layer
            'student_projection.2'   # Last projection layer
        ]
        
        for layer in key_layers:
            if layer in self.activations:
                feature_map = self.visualize_feature_maps(layer)
                if feature_map:
                    feature_map_plots.append((layer, feature_map))
        
        # Create an HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DINO Model Debug Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .image-container {{ margin-top: 10px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>DINO Model Debug Report</h1>
            <div class="section">
                <h2>Training Progress</h2>
                <div class="image-container">
                    <h3>Loss Curve</h3>
                    <img src="{loss_plot.name}" alt="Loss Curve">
                </div>
                <div class="image-container">
                    <h3>Student-Teacher Similarity</h3>
                    <img src="{similarity_plot.name if similarity_plot else 'none'}" alt="Student-Teacher Similarity">
                </div>
            </div>
            
            <div class="section">
                <h2>Gradient Analysis</h2>
                <div class="image-container">
                    <h3>Gradient Norms</h3>
                    <img src="{grad_norm_plot.name}" alt="Gradient Norms">
                </div>
                <div class="image-container">
                    <h3>Gradient Flow</h3>
                    <img src="{grad_flow_plot.name if grad_flow_plot else 'none'}" alt="Gradient Flow">
                </div>
            </div>
            
            <div class="section">
                <h2>Feature Representation</h2>
                <div class="image-container">
                    <h3>t-SNE Visualization of Embeddings</h3>
                    <img src="{tsne_plot.name if tsne_plot else 'none'}" alt="t-SNE Visualization">
                </div>
                <div class="image-container">
                    <h3>Weight Distributions</h3>
                    <img src="{weight_dist_plot.name}" alt="Weight Distributions">
                </div>
            </div>
            
            <div class="section">
                <h2>Model Health</h2>
                <div class="image-container">
                    <h3>Dead Neurons</h3>
                    <img src="{dead_neurons_plot.name if dead_neurons_plot else 'none'}" alt="Dead Neurons">
                </div>
                <div class="image-container">
                    <h3>Batch Statistics</h3>
                    <img src="{batch_stats_plot.name}" alt="Batch Statistics">
                </div>
            </div>

            <div class="section">
                <h2>Center Updates</h2>
                <div class="image-container">
                    <h3>Center Values</h3>
                    <img src="{center_updates_plot.name}" alt="Center Values">
                </div>
            </div>
            
            <div class="section">
                <h2>Feature Maps</h2>
        """
        
        for layer, plot in feature_map_plots:
            html_content += f"""
                <div class="image-container">
                    <h3>Feature Map: {layer}</h3>
                    <img src="{plot.name}" alt="Feature Map for {layer}">
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        html_path = self.output_dir / "debug_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive debug report saved to {html_path}")
        return html_path


# Usage function to integrate with existing UniModalDINOLightning
def add_debugging_to_lightning_module(model_class):
    """
    Extend the pytorch lightning module with debugging capabilities
    
    Args:
        model_class: The lightning module class to extend
    
    Returns:
        Extended lightning module class
    """
    class DebugEnabledDINO(model_class):
        def __init__(self, *args, debug_dir="debug_outputs", **kwargs):
            super().__init__(*args, **kwargs)
            self.debug_dir = debug_dir
            self.debugger = ModelDebugger(self.model, output_dir=debug_dir)
            self.debug_frequency = 1  # Run debug every N epochs
            
        def on_train_epoch_start(self):
            super().on_train_epoch_start()
            if self.current_epoch % self.debug_frequency == 0:
                self.debugger.register_hooks()
        
        def on_train_epoch_end(self):
            super().on_train_epoch_end()
            if self.current_epoch % self.debug_frequency == 0:
                # Compute average loss from the epoch
                avg_loss = self.trainer.callback_metrics['train_loss_epoch'].mean().item()
                self.debugger.log_epoch_metrics(self.current_epoch, avg_loss)
                
                # Generate debug visualizations if at the end of training
                if self.current_epoch == self.trainer.max_epochs - 1:
                    self.debugger.generate_comprehensive_report()
                
                # Remove hooks to avoid overhead in other epochs
                self.debugger.remove_hooks()
                
        def on_train_batch_end(self, outputs, batch, batch_idx):
            super().on_train_batch_end(outputs, batch, batch_idx)
            # For first batch of debug epochs, perform immediate visualizations
            if self.current_epoch % self.debug_frequency == 0 and batch_idx == 0:
                # Store additional metrics for this batch
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    loss = outputs
                    
                self.debugger.log_epoch_metrics(self.current_epoch, loss.item())
    
    return DebugEnabledDINO