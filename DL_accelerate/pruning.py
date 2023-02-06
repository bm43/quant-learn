import keras
import torch
import torch.nn as nn
import torch.optim as optim

# to be updated when code is ready
# https://medium.com/@souvik.paul01/pruning-in-deep-learning-models-1067a19acd89

# train - prune - tune - prune - tune - ...

## post-train prunings:

# model pruning by removing layers that are "less important"
def structured_pruning(model: nn.Module):
    importance_threshold = 0.1
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            if torch.sum(weight.abs()) < importance_threshold:
                setattr(model, name, nn.Identity())
    return model

# model pruning that removes "less important" neurons based on abs(weight), = magnitude pruning
def targeted_pruning(model: nn.Module):
    importance_threshold = 0.1
    for name, module in model.named_parameters():
        if 'weight' in name:
            weight = module.data
            mask = (weight.abs() >= importance_threshold).float()
            module.data *= mask
    return model

# OBD
def optimized_brain_damage(model: nn.Module):    
    importance_threshold = 0.1

    for name, module in model.named_parameters():
        if 'weight' in name:
            weight = module.data
            # Save the original weight values
            original_weights = weight.clone()
            
            # Remove the current weights and calculate the accuracy of the model
            weight[:] = 0
            accuracy = calculate_accuracy(model, x_test, y_test)
            
            # Restore the original weights and calculate the accuracy of the model
            weight[:] = original_weights
            accuracy_with_weights = calculate_accuracy(model, x_test, y_test)
            
            # Calculate the difference in accuracy
            accuracy_diff = accuracy_with_weights - accuracy
            
            # If the difference in accuracy is below the threshold, set the weight to 0
            if accuracy_diff < importance_threshold:
                weight[:] = 0
    return model

    def lottery_ticket(model: nn.Module):
        winning_ticket = []
        for name, module in model.named_parameters():
            if 'weight' in name:
                weight = module.data
                # Find the indices of the weights with high magnitude
                high_magnitude_indices = torch.abs(weight) > 0.1
                # Save the high magnitude weights
                winning_ticket.append((name, weight[high_magnitude_indices].clone()))

        # Create a new model with only the winning ticket weights
        new_model = Net()
        for name, module in new_model.named_parameters():
            if 'weight' in name:
                for winning_ticket_name, winning_ticket_weights in winning_ticket:
                    if name == winning_ticket_name:
                        module.data[:] = winning_ticket_weights

        return new_model