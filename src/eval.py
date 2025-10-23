import torch
from typing import Dict, List, Tuple
import json

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def comprehensive_evaluation(
    model: torch.nn.Module,
    data_manager,
    device: torch.device,
    num_classes: int,
    increment: int,
    criterion: torch.nn.Module,
    save_dir: str = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the fully trained model on all previous tasks.
    
    Args:
        model: The fully trained model
        data_manager: Dataset manager
        device: Device to run evaluation on
        num_classes: Total number of classes
        increment: Number of classes per task
        criterion: Loss function
        save_dir: Optional directory to save evaluation results
        
    Returns:
        Dictionary with task-wise evaluation results
    """
    model.eval()
    results = {}
    
    print("=" * 50)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 50)
    
    # Evaluate on each task
    for task_idx in range(0, num_classes, increment):
        task_num = task_idx // increment + 1
        task_classes = list(range(task_idx, min(task_idx + increment, num_classes)))
        
        print(f"\nEvaluating Task {task_num} (Classes: {task_classes})")
        
        # Get test data for this task
        test_loader = data_manager.get_loader(
            mode='test', 
            label=task_classes, 
            batch_size=64, 
            shuffle=False
        )
        
        # Evaluate on this task
        task_loss, task_accuracy = evaluate(model, test_loader, criterion, device)
        
        results[f"task_{task_num}"] = {
            "classes": task_classes,
            "loss": task_loss,
            "accuracy": task_accuracy,
            "num_samples": len(test_loader.dataset)
        }
    
    # Calculate overall statistics
    all_accuracies = [result["accuracy"] for result in results.values()]
    all_losses = [result["loss"] for result in results.values()]
    
    overall_stats = {
        "mean_accuracy": sum(all_accuracies) / len(all_accuracies),
        "std_accuracy": torch.tensor(all_accuracies).std().item(),
        "mean_loss": sum(all_losses) / len(all_losses),
        "std_loss": torch.tensor(all_losses).std().item(),
        "num_tasks": len(results)
    }
    
    results["overall"] = overall_stats
    
    print("\n" + "=" * 50)
    print("OVERALL STATISTICS")
    print("=" * 50)
    print(f"Mean Accuracy: {overall_stats['mean_accuracy']:.2f}% ± {overall_stats['std_accuracy']:.2f}%")
    print(f"Mean Loss: {overall_stats['mean_loss']:.4f} ± {overall_stats['std_loss']:.4f}")
    print(f"Number of Tasks: {overall_stats['num_tasks']}")
    
    # Save results if save_dir is provided
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, "comprehensive_evaluation.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    return results