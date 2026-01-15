# illustrates the concept of quantization

import numpy as np

# Define the quantization function
def quantize(x, n_bits):
    max_val = np.max(np.abs(x))
    scale_factor = 2**(n_bits-1) / max_val
    x = np.round(x * scale_factor)
    x = np.clip(x, -2**(n_bits-1), 2**(n_bits-1)-1)
    return x

# Define the dequantization function
def dequantize(x, n_bits):
    max_val = np.max(np.abs(x))
    scale_factor = 2**(n_bits-1) / max_val
    x = x / scale_factor
    return x

"""
# using these with torch.nn would look like:

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Quantize the model's parameters
model.fc1_weights = quantize(model.fc1_weights, 8)
model.fc1_bias = quantize(model.fc1_bias, 8)
model.fc2_weights = quantize(model.fc2_weights, 8)
model.fc2_bias = quantize(model.fc2_bias, 8)

# Training loop
for epoch in range(5):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        # Dequantize the inputs
        inputs = dequantize(inputs, 8)
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Quantize the model's parameters
        model.fc1_weights = quantize(model.fc1_weights, 8)
        model.fc1_bias = quantize(model.fc1_bias, 8)
        model.fc2_weights = quantize(model.fc2_weights, 8)
        model.fc2_bias = quantize(model.fc2_bias, 8)

# training loop with quantization at the same time:
for epoch in range(5):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        # Dequantize the inputs
        inputs = dequantize(inputs, 8)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Quantize the model's parameters
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.data = quantize(param.data, 8)
"""