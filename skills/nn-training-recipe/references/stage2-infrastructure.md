# Stage 2: Training Infrastructure Setup

## Goal

Build a complete, verifiable training pipeline with the simplest possible model. Every component should be tested before adding complexity.

## Infrastructure Components Checklist

### 1. Reproducibility Setup

**Fix all sources of randomness**:
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For complete reproducibility (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

**Why**: Without fixed seeds, you can't tell if performance changes come from code changes or random variation.

### 2. Data Pipeline Verification

**Start simple - disable augmentation initially**:
```python
# BAD - start with complex augmentation
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
])

# GOOD - start with minimal processing
transform = transforms.Compose([
    transforms.ToTensor(),
])
```

**Visualize what enters the network**:
```python
# Display a batch of training data
def show_batch(dataloader):
    batch = next(iter(dataloader))
    images, labels = batch

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            # Denormalize if needed
            img = images[idx].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.set_title(f"Label: {labels[idx].item()}")
            ax.axis('off')
    plt.show()

show_batch(train_loader)
```

**Why**: Catches preprocessing bugs (wrong normalization, incorrect labels, corrupted images).

### 3. Model Initialization Checks

**Verify loss at initialization**:
```python
model = YourModel()
criterion = nn.CrossEntropyLoss()

# Get one batch
images, labels = next(iter(train_loader))
outputs = model(images)
initial_loss = criterion(outputs, labels)

print(f"Initial loss: {initial_loss.item():.4f}")

# For C-class classification with CrossEntropyLoss
# Expected: -log(1/C) = log(C)
num_classes = outputs.size(1)
expected_loss = np.log(num_classes)
print(f"Expected loss: {expected_loss:.4f}")

if abs(initial_loss.item() - expected_loss) > 0.5:
    print("WARNING: Initial loss differs significantly from expected!")
```

**Common expected losses**:
- CrossEntropy with C classes: `log(C)`
  - 10 classes: ~2.30
  - 100 classes: ~4.61
  - 1000 classes: ~6.91
- Binary CrossEntropy: `log(2) ≈ 0.693`
- MSE regression (normalized targets): depends on target distribution

**Why**: Wrong initialization suggests bugs in loss computation, model architecture, or data preprocessing.

### 4. Single Batch Overfitting Test

**The most important sanity check**:
```python
# Get a tiny batch (2-8 examples)
tiny_batch = next(iter(train_loader))
images, labels = tiny_batch
images = images[:4]  # Just 4 examples
labels = labels[:4]

# Train only on this batch
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for step in range(1000):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean()
            print(f"Step {step}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")

# Should reach loss ≈ 0 and accuracy = 1.0
```

**Success criteria**: Loss near zero, 100% accuracy on the tiny batch within a few hundred steps.

**If this fails**:
- Learning rate too low → increase by 10x
- Model too small → add capacity
- Optimizer issue → try SGD with momentum instead
- Bug in forward pass → check model architecture
- Bug in loss → verify loss computation
- Bug in data → check labels and inputs

### 5. Simple Baseline Model

**Start with the simplest model that could possibly work**:

```python
# For images: Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# For text: Simple LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x
```

**Why simple baselines**:
- Faster iteration during debugging
- Easier to understand what's going wrong
- Often competitive with complex models when data is limited
- Can always add complexity later in Stage 5

### 6. Training Loop with Verification

**Minimal training loop with proper logging**:
```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Check for NaN gradients
        if torch.isnan(loss):
            print("WARNING: NaN loss detected!")
            break

        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
```

### 7. Visualization and Logging

**Track metrics over time**:
```python
import matplotlib.pyplot as plt

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss over Time')

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy over Time')

    plt.tight_layout()
    plt.show()
```

## Common Infrastructure Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss is NaN | Learning rate too high | Reduce LR by 10x |
| Loss is NaN | Numerical instability | Use gradient clipping |
| Loss doesn't decrease | Learning rate too low | Increase LR by 10x |
| Loss doesn't decrease | Optimizer not updating | Check if gradients are flowing |
| Validation worse than random | Data pipeline bug | Check labels match inputs |
| Can't overfit small batch | Model bug | Verify forward pass is correct |
| Slow training | Data loading bottleneck | Use `num_workers>0` in DataLoader |

## Infrastructure Validation Checklist

Before moving to Stage 3, verify:

- [ ] Fixed random seed for reproducibility
- [ ] Visualized actual inputs to the network
- [ ] Verified loss at initialization matches theory
- [ ] Successfully overfit a batch of 4-8 examples
- [ ] Training loop runs without errors
- [ ] Metrics are logged and visualized
- [ ] Can save and load model checkpoints
- [ ] Validation pipeline works correctly

Only proceed when ALL checks pass. Bugs here will waste weeks later.
