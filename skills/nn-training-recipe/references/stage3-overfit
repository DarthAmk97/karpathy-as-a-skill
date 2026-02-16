# Stage 3: Overfitting

## Goal

Prove that your model architecture and training setup can achieve near-perfect performance on the training set. If you can't overfit, you have a fundamental bug that must be fixed before proceeding.

## Why Overfitting is Essential

**Key insight**: Overfitting is not the enemy at this stage - it's the proof that your setup works.

- **Can't overfit** → Something is fundamentally broken
- **Can overfit** → Your model has sufficient capacity and your training pipeline works

Fix bugs now while the setup is simple, not later when regularization hides the symptoms.

## Overfitting Strategy

### 1. Scale Up Model Capacity

If your model can't overfit the full training set, make it larger:

**For CNNs**:
```python
# Too small - might not have enough capacity
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 6 * 6, num_classes)
    # ...

# Better - more capacity
class LargerCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    # ...
```

**For transformers**:
```python
# Increase model size
model = Transformer(
    d_model=512,      # Was 256
    nhead=8,          # Was 4
    num_layers=6,     # Was 3
    dim_feedforward=2048,  # Was 1024
)
```

**For tabular MLPs**:
```python
# Add more layers and width
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
```

**Rule of thumb**: Use 2-5x more capacity than you think you'll need in production.

### 2. Train Longer

Sometimes the model can overfit, it just needs more iterations:

```python
# Train for many epochs with no early stopping
num_epochs = 200  # Or even 500+
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)

    # Log training performance
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

    # Check if we've overfit yet
    if train_acc > 0.99:
        print(f"Successfully overfit at epoch {epoch}")
        break
```

### 3. Reduce Regularization to Zero

At this stage, actively prevent regularization:

```python
# Remove ALL regularization
model = YourModel(
    dropout=0.0,      # No dropout
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0  # No weight decay
)

# No data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    # No random flips, crops, color jitter, etc.
])
```

### 4. Use Aggressive Learning Rate

Don't be conservative - you want to overfit quickly:

```python
# Try higher learning rates
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # Was 1e-3

# Or even adaptive approach
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
```

If training is unstable (loss explodes), reduce LR by 3-5x and try again.

## Debugging When You Can't Overfit

### Progressive Debugging Strategy

**Level 1: Single batch overfitting**
```python
# Verified in Stage 2, but double-check
tiny_batch = next(iter(train_loader))
images, labels = tiny_batch[:4]  # 4 examples

# Train on just these 4 examples
# Should reach 100% accuracy within minutes
```

**Level 2: Small subset (100 examples)**
```python
# Create small training subset
small_train = torch.utils.data.Subset(train_dataset, range(100))
small_loader = DataLoader(small_train, batch_size=32)

# Train on 100 examples
# Should reach >95% accuracy within 10-20 epochs
```

**Level 3: Larger subset (1000 examples)**
```python
medium_train = torch.utils.data.Subset(train_dataset, range(1000))
medium_loader = DataLoader(medium_train, batch_size=32)

# Train on 1000 examples
# Should reach >90% accuracy within 50 epochs
```

**Level 4: Full training set**
```python
# Now train on full dataset
# Target: >95% training accuracy
```

### Common Overfitting Blockers

| Issue | Diagnostic | Solution |
|-------|-----------|----------|
| Loss decreases but accuracy stuck | Optimization works but model wrong | Check loss function matches task |
| Loss plateaus early | Insufficient capacity | Increase model size |
| Loss oscillates | Learning rate too high | Reduce LR by 3-10x |
| Loss is NaN | Numerical instability | Reduce LR, add gradient clipping |
| Very slow progress | Learning rate too low | Increase LR by 3-10x |
| Gradients are zero | Dying ReLU or vanishing grads | Use LeakyReLU or check initialization |
| Memory error | Model too large | Reduce batch size or model size |

### Gradient Flow Diagnostics

If you're stuck, check if gradients are flowing:

```python
# After loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm={grad_norm:.6f}")
    else:
        print(f"{name}: NO GRADIENT")
```

**What to look for**:
- Gradients should be non-zero throughout the network
- Gradients shouldn't be too small (< 1e-6) or too large (> 1e2)
- If early layers have zero gradients, you have vanishing gradients
- If gradients are NaN, you have exploding gradients

### Learning Rate Sweep

Still stuck? Try a learning rate sweep:

```python
learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

for lr in learning_rates:
    model = YourModel()  # Fresh model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train for 10 epochs
    best_train_loss = float('inf')
    for epoch in range(10):
        train_loss, train_acc = train_epoch(...)
        best_train_loss = min(best_train_loss, train_loss)

    print(f"LR={lr}: Best train loss={best_train_loss:.4f}")
```

Choose the LR that gives the fastest decrease in training loss.

## Success Criteria

You've successfully completed Stage 3 when:

- [ ] Training loss near zero (< 0.1 for classification, < 1e-3 for regression)
- [ ] Training accuracy > 95% (or appropriate metric for your task)
- [ ] Model confidently predicts training examples correctly
- [ ] Validation performance is worse than training (this is expected!)

**Key verification**:
```python
# Check training set performance
train_loss, train_acc = validate(model, train_loader, criterion, device)
print(f"Final training: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

if train_acc < 0.95:
    print("WARNING: Haven't successfully overfit yet!")
    print("Don't proceed to Stage 4 until training accuracy is high")
```

## Mental Model

Think of overfitting as a "proof of concept":
- **Stage 2**: Proved the training loop runs
- **Stage 3**: Proved the model can learn patterns in your data
- **Stage 4**: Will tune how well it generalizes

Without Stage 3, you're trying to tune generalization on a model that might have fundamental bugs. Always establish that perfect training performance is achievable first.

## Common Mistakes

1. **Moving to regularization too early**: "My training accuracy is only 80% but validation is 75%, so I'll add regularization" → No! First get training to 99%.

2. **Blaming the data**: "Maybe my data is just hard to fit" → If humans can do the task, a neural network can overfit it.

3. **Using too small a model**: "I want it to generalize so I'll keep it small" → Capacity is free during overfitting. Add regularization in Stage 4, not here.

4. **Giving up too soon**: "I trained for 50 epochs and it's not overfitting" → Try 200 epochs, or 500. Once you confirm it CAN overfit, you can optimize training speed.

## Next Steps

Once training performance is excellent (>95% accuracy), you've proven:
- ✅ Your model architecture can represent the patterns in your data
- ✅ Your optimizer can find good parameters
- ✅ Your training loop is bug-free
- ✅ Your data pipeline is correct

Now you can confidently move to Stage 4 (Regularization) to improve generalization.
