# Stage 6: Squeeze Out the Juice

**Goal**: Extract final performance gains through ensemble methods, extended training, and test-time techniques.

## Core Philosophy

You've built a good model, tuned hyperparameters, and achieved solid performance. This stage is about squeezing out the last few percentage points of performance through techniques that add complexity but provide incremental gains.

**Key principle**: Only apply these techniques when the marginal gains are worth the engineering cost and computational expense.

**Warning**: This stage has diminishing returns. Don't spend weeks here unless you're in a competition or production system where every 0.1% matters.

---

## Technique 1: Model Ensembles

**Why ensembles work**: Different models make different errors. Combining predictions reduces variance and improves robustness.

### Types of Ensembles

#### 1. Different Random Seeds
**Simplest and most effective approach**

```python
# Train multiple models with different initializations
seeds = [42, 123, 456, 789, 101112]
models = []

for seed in seeds:
    torch.manual_seed(seed)
    model = MyModel().to(device)
    trained_model = train(model, train_loader, val_loader)
    models.append(trained_model)

# Ensemble prediction
def ensemble_predict(models, input_data):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(input_data)
            predictions.append(pred)

    # Average predictions
    return torch.stack(predictions).mean(dim=0)
```

**Typical gains**: 0.5-2% improvement
**Cost**: 5x training time for 5 models
**When to use**: Almost always worth it for important models

#### 2. Different Architectures
**Train models with different structures**

```python
# Different architectures
models = [
    ResNet50(),
    EfficientNet(),
    DenseNet121(),
    VisionTransformer()
]

# Train each and ensemble
# More diversity often means better ensemble performance
```

**Typical gains**: 1-3% improvement
**Cost**: High - need to tune each architecture separately
**When to use**: Competitions, production systems where performance is critical

#### 3. Different Training Data Splits
**K-fold cross-validation ensembles**

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_seed=42)
models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    model = MyModel()
    trained_model = train(model, train_subset, val_subset)
    models.append(trained_model)

# All models trained on different data splits
```

**Typical gains**: 0.5-1.5% improvement
**Cost**: K-fold training time
**When to use**: Small datasets where data efficiency matters

#### 4. Different Stages of Training
**Snapshot ensembles** (free ensembles from one training run)

```python
# Save models at different epochs
# Use cyclic learning rate to get models at different local minima

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
snapshots = []

for epoch in range(epochs):
    train_epoch(model, train_loader, optimizer)
    scheduler.step()

    # Save snapshot at end of each cycle
    if (epoch + 1) % 10 == 0:
        snapshots.append(copy.deepcopy(model))
```

**Typical gains**: 0.3-1% improvement
**Cost**: Minimal - just one training run
**When to use**: When training time is constrained

### Ensemble Combination Strategies

#### Simple Averaging (Most Common)
```python
# For classification probabilities
ensemble_probs = sum(model.predict_proba(X) for model in models) / len(models)

# For regression
ensemble_pred = sum(model.predict(X) for model in models) / len(models)
```

#### Weighted Averaging
```python
# Weight by validation performance
weights = [0.25, 0.30, 0.20, 0.25]  # Better models get higher weight

ensemble_pred = sum(w * model.predict(X) for w, model in zip(weights, models))
```

#### Voting (Classification)
```python
# Hard voting - majority vote
from scipy.stats import mode

predictions = [model.predict(X) for model in models]
ensemble_pred = mode(predictions, axis=0)[0]
```

#### Stacking (Advanced)
```python
# Train a meta-model on predictions from base models
# Base model predictions become features for meta-model

from sklearn.linear_model import LogisticRegression

# Get predictions from base models on validation set
base_predictions = np.column_stack([
    model.predict_proba(X_val) for model in models
])

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(base_predictions, y_val)

# Final prediction
test_base_predictions = np.column_stack([
    model.predict_proba(X_test) for model in models
])
ensemble_pred = meta_model.predict(test_base_predictions)
```

---

## Technique 2: Test-Time Augmentation (TTA)

**Idea**: Generate multiple augmented versions of each test input, predict on all versions, and average the predictions.

### Image Classification Example
```python
import torchvision.transforms as transforms

def test_time_augmentation(model, image, n_augmentations=5):
    """Apply TTA for image classification."""

    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])

    predictions = []
    model.eval()

    # Original prediction
    with torch.no_grad():
        predictions.append(model(image))

    # Augmented predictions
    for _ in range(n_augmentations - 1):
        augmented = augmentations(image)
        with torch.no_grad():
            predictions.append(model(augmented))

    # Average all predictions
    return torch.stack(predictions).mean(dim=0)
```

### Deterministic TTA (Preferred)
```python
def deterministic_tta(model, image):
    """Apply specific, deterministic augmentations."""

    predictions = []

    # Original
    predictions.append(model(image))

    # Horizontal flip
    predictions.append(model(transforms.functional.hflip(image)))

    # Vertical flip
    predictions.append(model(transforms.functional.vflip(image)))

    # Rotations
    for angle in [90, 180, 270]:
        rotated = transforms.functional.rotate(image, angle)
        predictions.append(model(rotated))

    return torch.stack(predictions).mean(dim=0)
```

**Typical gains**: 0.5-2% improvement
**Cost**: Inference time increases by N times (N augmentations)
**When to use**: When inference time is not critical, or for important samples only

---

## Technique 3: Extended Training

**Simple but effective**: Train longer once you know your hyperparameters work.

### When to Train Longer

1. **You have spare compute**: Training is running anyway, let it continue
2. **Loss is still decreasing**: Model hasn't converged yet
3. **Large dataset**: More data needs more epochs to see all examples
4. **Strong regularization**: Prevents overfitting even with longer training

### How Much Longer?

```python
# Rule of thumb: If you trained for N epochs, try 2N or 3N

# Original training
epochs = 50
# Extended training
epochs = 150  # 3x longer

# Watch validation loss - if it plateaus or increases, stop
```

### Learning Rate Warmup for Extended Training

```python
# When training longer, consider learning rate warmup
# Helps with training stability

def warmup_scheduler(epoch, warmup_epochs=10, base_lr=1e-3, warmup_start_lr=1e-6):
    if epoch < warmup_epochs:
        # Linear warmup
        return warmup_start_lr + (base_lr - warmup_start_lr) * epoch / warmup_epochs
    else:
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

**Typical gains**: 0.5-2% improvement
**Cost**: Proportional to training time extension
**When to use**: When you have compute budget and model is still improving

---

## Technique 4: Fine-Tuning Strategies

### Progressive Unfreezing
**For transfer learning**: Gradually unfreeze layers during training

```python
# Start with frozen base, only train head
for param in model.base.parameters():
    param.requires_grad = False

# Train head for a few epochs
train(model, epochs=10)

# Unfreeze last few layers
for param in model.base.layer4.parameters():
    param.requires_grad = True

# Train with lower learning rate
train(model, epochs=10, lr=1e-5)

# Unfreeze entire model
for param in model.base.parameters():
    param.requires_grad = True

# Final fine-tuning with very low learning rate
train(model, epochs=10, lr=1e-6)
```

### Discriminative Learning Rates
**Different learning rates for different layers**

```python
# Lower learning rates for earlier layers (already pretrained)
# Higher learning rates for later layers (need more adaptation)

optimizer = torch.optim.Adam([
    {'params': model.base.layer1.parameters(), 'lr': 1e-5},
    {'params': model.base.layer2.parameters(), 'lr': 3e-5},
    {'params': model.base.layer3.parameters(), 'lr': 1e-4},
    {'params': model.base.layer4.parameters(), 'lr': 3e-4},
    {'params': model.head.parameters(), 'lr': 1e-3}
])
```

**Typical gains**: 1-3% improvement for transfer learning
**Cost**: More complex training code
**When to use**: When using pretrained models on new domains

---

## Technique 5: Prediction Calibration

**Problem**: Model outputs might not be well-calibrated (confidence doesn't match accuracy)

### Temperature Scaling
```python
class TemperatureScaling(nn.Module):
    """Calibrate model predictions with temperature."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

# Optimize temperature on validation set
calibrated_model = TemperatureScaling(model)
optimizer = torch.optim.LBFGS([calibrated_model.temperature], lr=0.01, max_iter=50)

def calibration_loss():
    optimizer.zero_grad()
    logits = calibrated_model(val_inputs)
    loss = F.cross_entropy(logits, val_targets)
    loss.backward()
    return loss

optimizer.step(calibration_loss)

print(f"Optimal temperature: {calibrated_model.temperature.item():.3f}")
```

### Isotonic Regression (More Flexible)
```python
from sklearn.isotonic import IsotonicRegression

# Get predictions on validation set
val_probs = model.predict_proba(X_val)[:, 1]  # For binary classification

# Fit calibrator
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_probs, y_val)

# Apply to test set
test_probs = model.predict_proba(X_test)[:, 1]
calibrated_probs = calibrator.transform(test_probs)
```

**Typical gains**: Better confidence estimates, same accuracy
**When to use**: When probability estimates matter (medical, finance applications)

---

## Technique 6: Self-Training / Pseudo-Labeling

**Idea**: Use model predictions on unlabeled data as additional training data

```python
def self_training(model, labeled_data, unlabeled_data, confidence_threshold=0.95):
    """Iteratively add high-confidence predictions to training set."""

    # Train on labeled data
    train(model, labeled_data)

    # Predict on unlabeled data
    model.eval()
    pseudo_labels = []
    pseudo_data = []

    with torch.no_grad():
        for inputs in unlabeled_data:
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_prob, predicted = probs.max(dim=1)

            # Keep only high-confidence predictions
            high_conf_mask = max_prob > confidence_threshold
            pseudo_data.append(inputs[high_conf_mask])
            pseudo_labels.append(predicted[high_conf_mask])

    # Combine labeled and pseudo-labeled data
    combined_data = combine(labeled_data, pseudo_data, pseudo_labels)

    # Retrain on combined dataset
    train(model, combined_data)

    return model
```

**Typical gains**: 1-5% improvement with large unlabeled datasets
**Cost**: Additional training iterations
**When to use**: Semi-supervised learning scenarios with lots of unlabeled data

---

## Technique 7: Knowledge Distillation

**Idea**: Train a smaller "student" model to mimic a larger "teacher" model

```python
def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.5):
    """
    Combined loss for knowledge distillation.

    Args:
        temperature: Softens probability distributions
        alpha: Weight between distillation loss and true label loss
    """
    # Distillation loss (KL divergence between teacher and student)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
    distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
    distillation_loss *= (temperature ** 2)

    # Standard cross-entropy loss with true labels
    student_loss = F.cross_entropy(student_logits, true_labels)

    # Combined loss
    return alpha * distillation_loss + (1 - alpha) * student_loss

# Training loop
teacher_model.eval()  # Teacher is frozen
student_model.train()

for inputs, labels in train_loader:
    # Get teacher predictions
    with torch.no_grad():
        teacher_logits = teacher_model(inputs)

    # Student forward pass
    student_logits = student_model(inputs)

    # Compute combined loss
    loss = distillation_loss(student_logits, teacher_logits, labels)

    # Optimize student
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Typical gains**: Smaller model with 90-95% of large model performance
**When to use**: Deploying to resource-constrained environments (mobile, edge)

---

## Putting It All Together

### Decision Tree for Stage 6 Techniques

```
Start
  |
  ├─ Do you need better accuracy at any cost?
  |    └─ YES: Use model ensembles (5-10 models) + TTA
  |    └─ NO: Continue
  |
  ├─ Is inference time critical?
  |    └─ YES: Skip TTA, use single model or small ensemble
  |    └─ NO: Consider TTA
  |
  ├─ Do you have spare compute?
  |    └─ YES: Train longer, try extended epochs
  |    └─ NO: Use current training
  |
  ├─ Are you using transfer learning?
  |    └─ YES: Try progressive unfreezing, discriminative LR
  |    └─ NO: Continue
  |
  ├─ Do probability estimates need to be accurate?
  |    └─ YES: Apply calibration (temperature scaling)
  |    └─ NO: Skip calibration
  |
  ├─ Do you have lots of unlabeled data?
  |    └─ YES: Try self-training / pseudo-labeling
  |    └─ NO: Skip
  |
  └─ Need to deploy to resource-constrained device?
       └─ YES: Use knowledge distillation
       └─ NO: Done!
```

### Example Production Pipeline

```python
def create_production_model(train_data, val_data, test_data):
    """Full pipeline for production model."""

    # 1. Train ensemble of models with different seeds
    print("Training ensemble...")
    models = []
    for seed in [42, 123, 456, 789, 101112]:
        model = train_model(train_data, val_data, seed=seed, epochs=150)
        models.append(model)

    # 2. Apply temperature scaling for calibration
    print("Calibrating predictions...")
    calibrated_models = []
    for model in models:
        calibrated = calibrate_model(model, val_data)
        calibrated_models.append(calibrated)

    # 3. Create ensemble predictor with TTA
    def production_predict(input_data, use_tta=True):
        if use_tta:
            # Apply test-time augmentation
            predictions = []
            for model in calibrated_models:
                tta_pred = test_time_augmentation(model, input_data, n_aug=5)
                predictions.append(tta_pred)
        else:
            # Direct prediction
            predictions = [model(input_data) for model in calibrated_models]

        # Ensemble average
        return torch.stack(predictions).mean(dim=0)

    # 4. Evaluate on test set
    test_accuracy = evaluate(production_predict, test_data)
    print(f"Final test accuracy: {test_accuracy:.3f}")

    return production_predict

# Use in production
production_model = create_production_model(train_data, val_data, test_data)
```

---

## Performance Benchmarks

**Expected cumulative gains from Stage 6 techniques**:

| Technique | Typical Improvement | Computational Cost |
|-----------|---------------------|-------------------|
| Ensemble (5 models) | +1.0% to +2.0% | 5x training, 5x inference |
| Test-Time Augmentation | +0.5% to +1.5% | Nx inference (N augmentations) |
| Extended Training (3x) | +0.5% to +1.5% | 3x training time |
| Progressive Unfreezing | +1.0% to +3.0% | Minimal (transfer learning only) |
| Temperature Calibration | Same accuracy, better probabilities | Negligible |
| Self-Training | +1.0% to +5.0% | 1-2x training iterations |
| Knowledge Distillation | -5% to -10% (smaller model) | Student training time |

**Stacking multiple techniques**:
- Ensemble + TTA: +1.5% to +3.5%
- Ensemble + Extended Training: +2.0% to +4.0%
- All techniques: +3.0% to +6.0% (with 10x cost)

---

## When to Stop

**Signs you should stop optimizing**:

1. **Cost exceeds value**: Another 0.1% improvement requires 10x more compute
2. **Hitting fundamental limits**: You're at human performance or Bayes error
3. **Production constraints**: Model is too large/slow for deployment requirements
4. **Time constraints**: Deadline approaching, current model is "good enough"
5. **Diminishing returns**: Last 5 techniques gave <0.1% improvement each

**Remember**: Perfect is the enemy of good. Ship a working model, then iterate.

---

## Summary Checklist

Final optimization checklist:

- [ ] Have I trained an ensemble of at least 3-5 models?
- [ ] Have I tried test-time augmentation on validation set?
- [ ] Have I trained models longer to check for continued improvement?
- [ ] (If using transfer learning) Have I tried progressive unfreezing?
- [ ] Have I calibrated predictions if probability estimates matter?
- [ ] (If applicable) Have I tried self-training with unlabeled data?
- [ ] Have I evaluated the cost-benefit of each technique?
- [ ] Have I measured final performance on a held-out test set?
- [ ] Have I documented the final model architecture and techniques used?
- [ ] Is the model ready for production deployment?

If yes to relevant items, **congratulations** - you've successfully completed the neural network training recipe!

---

## Final Thoughts

Stage 6 is about squeezing out the last drops of performance. These techniques work, but come with significant costs in complexity, computation, and maintenance.

**Golden rule**: Only apply these techniques when the gains justify the costs. For most projects, a well-tuned single model from Stages 1-5 is sufficient.

For competitions, research papers, or critical production systems where every percentage point matters, Stage 6 techniques can make the difference between good and great.

**Now go build something amazing!** 🚀
