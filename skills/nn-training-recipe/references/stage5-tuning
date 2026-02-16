# Stage 5: Tune Hyperparameters

**Goal**: Find the best configuration through systematic exploration to maximize model performance.

## Core Philosophy

Hyperparameter tuning is about efficient exploration, not exhaustive search. Neural networks have complex, non-linear sensitivity to different parameters. Some hyperparameters (like learning rate) dramatically affect performance, while others have minimal impact.

**Key principle**: Get 80% of the gains from 20% of the effort. Don't over-optimize.

---

## Random Search vs Grid Search

**Use random search, not grid search.**

**Why random search wins**:
- Neural networks have varying parameter sensitivity
- Some parameters matter much more than others
- Random search explores more unique values for important parameters
- Grid search wastes computation on unimportant parameter combinations

**Example**:
```python
# BAD: Grid search
learning_rates = [0.001, 0.01, 0.1]
weight_decays = [0.0001, 0.001, 0.01]
# Only tests 9 combinations, limited exploration

# GOOD: Random search
import numpy as np
learning_rate = 10 ** np.random.uniform(-4, -1)  # Explores continuous range
weight_decay = 10 ** np.random.uniform(-5, -2)
```

---

## Hyperparameter Priority List

Focus your tuning effort on parameters in rough order of importance:

### 1. Learning Rate (Most Important)
- **Impact**: Often the single most important hyperparameter
- **Search strategy**: Log scale from 10^-5 to 10^-1
- **Signs of problems**:
  - Too high: Loss explodes or oscillates wildly
  - Too low: Loss decreases very slowly, training takes forever
- **Tips**: Start with 3e-4 as a reasonable default for Adam

### 2. Learning Rate Schedule
- **Common schedules**:
  - Constant (simplest baseline)
  - Step decay (drop LR at milestones)
  - Cosine annealing (smooth decay)
  - Exponential decay
  - One-cycle policy (warmup then decay)
- **When to use**: After finding good base LR, try scheduling for final gains
- **Example**:
  ```python
  # Cosine annealing with warmup
  from torch.optim.lr_scheduler import CosineAnnealingLR
  scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
  ```

### 3. Model Architecture
- **Dimensions to tune**:
  - Depth (number of layers)
  - Width (hidden dimensions, number of filters)
  - Architecture type (ResNet, Transformer, etc.)
- **Strategy**: Start simple, gradually increase capacity
- **Rule of thumb**: Bigger is often better, but regularize appropriately

### 4. Batch Size
- **Trade-offs**:
  - Larger batches: More stable gradients, faster training (GPU utilization)
  - Smaller batches: Better generalization, acts as regularization
- **Practical range**: 32-512 for most tasks
- **GPU memory**: Largest batch that fits in memory is often good
- **Note**: May need to adjust learning rate when changing batch size

### 5. Weight Decay (L2 Regularization)
- **Search range**: 10^-5 to 10^-2 (log scale)
- **Effect**: Prevents weights from growing too large
- **When to tune**: After you have a good learning rate
- **Typical values**: 1e-4 to 1e-3 for most problems

### 6. Dropout Rate
- **Range**: 0.0 to 0.5
- **Common values**: 0.1, 0.2, 0.3, 0.5
- **Where to apply**: Between fully connected layers, attention layers
- **Note**: Often less important with batch normalization

### Lower Priority Parameters
- Optimizer choice (Adam is usually fine)
- Batch normalization momentum
- Gradient clipping threshold
- Initialization schemes (defaults usually work)

---

## Coarse-to-Fine Strategy

**Phase 1: Broad exploration**
1. Define wide ranges for key hyperparameters
2. Run 20-50 random configurations
3. Identify promising regions
4. Look for patterns in what works

**Phase 2: Narrow refinement**
1. Focus ranges around best performers from Phase 1
2. Run another 20-30 configurations in tighter ranges
3. Find the best configuration

**Example workflow**:
```python
# Phase 1: Coarse search
coarse_ranges = {
    'learning_rate': (1e-5, 1e-1),  # Wide range
    'weight_decay': (1e-6, 1e-2),
    'hidden_dim': [128, 256, 512, 1024]
}

# After analysis, saw best results around lr=3e-4
# Phase 2: Fine search
fine_ranges = {
    'learning_rate': (1e-4, 1e-3),  # Narrow range
    'weight_decay': (1e-5, 1e-3),
    'hidden_dim': [256, 512]  # Focus on promising sizes
}
```

---

## What to Track

**Essential metrics**:
- Training loss
- Validation loss
- Training accuracy/metric
- Validation accuracy/metric
- Training time per epoch
- Total training time

**Useful additional tracking**:
- Learning rate over time (if using schedule)
- Gradient norms
- Weight norms
- Best epoch (for early stopping)
- GPU memory usage

**Tools**:
- MLflow (experiment tracking)
- Weights & Biases (visualization)
- TensorBoard (built into PyTorch/TF)
- Neptune.ai
- Aim

**Example with MLflow**:
```python
import mlflow

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        'learning_rate': lr,
        'batch_size': batch_size,
        'hidden_dim': hidden_dim
    })

    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch()
        val_loss = validate()

        # Log metrics
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss
        }, step=epoch)

    # Log final model
    mlflow.pytorch.log_model(model, 'model')
```

---

## Analysis and Visualization

After running experiments, analyze results to understand patterns:

### 1. Compare Top Performers
```python
# Look at your top 5 runs
# What do they have in common?
# What ranges work best?
```

### 2. Parameter Importance
- Which parameters show clear trends?
- Which have little effect on performance?
- Are there interactions between parameters?

### 3. Visualizations
- **Learning curves**: Plot train/val loss over time for different configs
- **Scatter plots**: Hyperparameter value vs final validation performance
- **Parallel coordinates**: Visualize multiple hyperparameters simultaneously

### 4. Identify Failure Modes
- Configurations that diverged (NaN loss)
- Configurations that converged too slowly
- Configurations that overfit severely

---

## Learning Rate Strategies

### Finding the Right Learning Rate

**Method 1: LR Range Test**
```python
# Gradually increase LR and plot loss
# Find the steepest decline in loss
# Use LR slightly before loss increases

from torch_lr_finder import LRFinder

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()  # Look for steepest descent
```

**Method 2: Start Small and Scale Up**
- Start with very small LR (1e-5)
- If loss decreases steadily but slowly, increase by 3-10x
- Keep scaling until you see instability
- Use the largest stable LR

### Adaptive Learning Rates

**Optimizer comparison**:
- **SGD**: Requires careful LR tuning, often needs scheduling
- **SGD + Momentum**: Better than vanilla SGD, still needs tuning
- **Adam**: Adaptive LR per parameter, good default choice
- **AdamW**: Adam with decoupled weight decay (often best)
- **RAdam**: Rectified Adam, more robust
- **Lookahead**: Wraps other optimizers, more stable

**Rule of thumb**: Start with Adam or AdamW with default params (lr=3e-4, betas=(0.9, 0.999))

---

## When to Stop Tuning

**Diminishing returns indicators**:
1. Last 10 runs show <1% improvement over best
2. You've explored the promising region thoroughly
3. Further tuning costs more than the performance gain is worth
4. You're hitting fundamental limits (e.g., label noise)

**Time budgets**:
- **Quick project**: 10-20 random runs
- **Standard project**: 50-100 runs
- **Competition/production**: 100-500+ runs with sophisticated methods

**Advanced methods** (only if worth the complexity):
- Bayesian optimization (e.g., Optuna, Ray Tune)
- Population-based training
- Hyperband (early stopping for bad configs)
- Neural architecture search (NAS)

---

## Common Pitfalls

### 1. Tuning Too Early
**Problem**: Tuning before verifying you can overfit
**Solution**: Complete Stage 3 (overfit) before extensive tuning

### 2. Not Using Log Scale
**Problem**: Searching LR in [0.001, 0.01, 0.1] linearly
**Solution**: Use 10^uniform(-4, -1) for multiplicative parameters

### 3. Changing Multiple Things at Once
**Problem**: Can't isolate what actually helped
**Solution**: Change one thing at a time when doing targeted experiments

### 4. Ignoring Variance
**Problem**: Running each config only once, getting misled by luck
**Solution**: Run best configs with multiple random seeds (3-5 seeds)

### 5. Overfitting to Validation Set
**Problem**: Tuning extensively on validation set, then it doesn't generalize to test
**Solution**: Use held-out test set only for final evaluation, not during tuning

### 6. Not Tracking Enough
**Problem**: Can't reproduce good results or understand why something worked
**Solution**: Log everything - hyperparameters, metrics, code versions, seeds

---

## Practical Workflow Example

```python
import mlflow
import numpy as np

def train_with_config(config, train_loader, val_loader, device):
    """Train model with given hyperparameters."""

    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])

    # Initialize model
    model = MyModel(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    # Optimizer with hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        # Log metrics
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

def random_search(n_trials=50):
    """Run random hyperparameter search."""

    mlflow.set_experiment("my_tuning_experiment")

    for trial in range(n_trials):
        # Sample random hyperparameters
        config = {
            'learning_rate': 10 ** np.random.uniform(-5, -2),
            'weight_decay': 10 ** np.random.uniform(-6, -3),
            'batch_size': np.random.choice([32, 64, 128, 256]),
            'hidden_dim': np.random.choice([128, 256, 512]),
            'num_layers': np.random.choice([2, 3, 4]),
            'dropout': np.random.uniform(0.0, 0.5),
            'epochs': 50,
            'seed': 42 + trial  # Different seed per trial
        }

        with mlflow.start_run():
            # Log all hyperparameters
            mlflow.log_params(config)

            # Train and get best validation loss
            best_val_loss = train_with_config(config, train_loader, val_loader, device)

            # Log final result
            mlflow.log_metric('best_val_loss', best_val_loss)

            print(f"Trial {trial+1}/{n_trials}: Val Loss = {best_val_loss:.4f}")

# Run the search
random_search(n_trials=50)
```

---

## Summary Checklist

Before moving to Stage 6:

- [ ] Have I found a learning rate that gives stable training?
- [ ] Have I tried a learning rate schedule?
- [ ] Have I explored different model sizes?
- [ ] Have I tested different batch sizes?
- [ ] Have I tuned regularization (weight decay, dropout)?
- [ ] Have I run top configs with multiple seeds to check variance?
- [ ] Have I logged all experiments for reproducibility?
- [ ] Am I seeing diminishing returns from further tuning?

If yes to most of these, proceed to Stage 6 for final optimizations.
