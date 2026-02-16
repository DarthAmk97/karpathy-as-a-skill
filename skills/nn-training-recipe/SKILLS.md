---
name: nn-training-recipe
description: Systematic 6-stage recipe for training neural networks based on Andrej Karpathy's methodology. MUST be invoked whenever the task is related to deep learning, training a model, debugging neural network training, or improving model performance. Use for: (1) Starting a new neural network training project, (2) Debugging training issues (loss not decreasing, poor performance, etc.), (3) Improving existing model performance, (4) Setting up training infrastructure, or (5) Any task involving neural network development and training.
---

# Neural Network Training Recipe

A systematic approach to training neural networks that avoids common pitfalls and silent failures. This recipe emphasizes patience, incremental complexity, and verification at every step.

## Core Philosophy

Neural networks are a "leaky abstraction" - they lack clean APIs and fail silently without errors, producing only degraded performance. Success requires methodical attention to detail, not fast-and-furious iteration.

**Key principle**: Build complexity incrementally while maintaining verifiable correctness throughout.

## The 6-Stage Recipe

### Stage 1: Become One With the Data

**Goal**: Develop deep intimacy with your dataset before writing any model code.

**Actions**:
1. **Manual inspection**: Look at thousands of examples (not just a few)
2. **Understand distributions**: What are the class balances? Feature ranges? Patterns?
3. **Find corrupted data**: Look for mislabeled examples, outliers, data quality issues
4. **Identify patterns**: What patterns are obvious to you? These guide architecture choices

**Time investment**: This is not a 10-minute task. Spend hours examining data.

**Why it matters**: Your intuition about the data informs every downstream decision - architecture, loss functions, data augmentation, and debugging strategies.

See [references/stage1-data.md](references/stage1-data.md) for detailed data exploration techniques.

---

### Stage 2: Set Up the End-to-End Training/Evaluation Skeleton

**Goal**: Build complete infrastructure with the simplest possible model before adding complexity.

**Actions**:
1. **Fix random seed**: Ensure reproducibility for debugging
2. **Simplify first**: Disable data augmentation initially
3. **Add verification**: Visualize inputs directly before they enter the network
4. **Use simple metrics**: Start with loss visualization before complex metrics
5. **Verify loss initialization**: Check that loss at init matches theoretical expectations
6. **Sanity check**: Overfit a tiny batch (2-8 examples) to verify the pipeline works

**Example loss initialization checks**:
- Softmax classification with C classes: `-log(1/C)` expected
- Binary classification: `-log(0.5) = 0.693` expected

**Red flags**:
- Can't overfit a single batch → fundamental bug in model or training loop
- Loss at init doesn't match theory → initialization or loss computation bug

See [references/stage2-infrastructure.md](references/stage2-infrastructure.md) for pipeline setup patterns.

---

### Stage 3: Overfit

**Goal**: Prove that your model can achieve perfect training performance.

**Actions**:
1. **Use more capacity**: Make the model larger than you think you need
2. **Train to zero loss**: Keep training until training loss is effectively zero
3. **Verify on one batch**: If struggling, first overfit a single batch, then scale up

**Why it matters**: If you can't overfit, you have a fundamental bug. Fix it now before adding regularization.

**Success criteria**: Training loss near zero (or perfect accuracy on training set)

See [references/stage3-overfit.md](references/stage3-overfit.md) for overfitting strategies and debugging.

---

### Stage 4: Regularize

**Goal**: Improve validation performance by adding constraints.

**Priority order**:
1. **Get more real data** (by far the best regularization)
2. **Data augmentation** (creative and domain-specific)
3. **Pretraining** (if applicable - use pretrained models when available)
4. **Simplify inputs** (reduce dimensionality, remove noisy features)
5. **Smaller model** (reduce capacity - fewer layers/filters/parameters)
6. **Reduce batch size** (can act as regularization)
7. **Dropout** (classic technique - apply to appropriate layers)
8. **Weight decay** (L2 regularization)
9. **Early stopping** (monitor validation loss)

**Key insight**: Engineering tricks come after data quality. Don't optimize batch normalization placement before trying to get more data.

See [references/stage4-regularize.md](references/stage4-regularize.md) for regularization techniques and when to use each.

---

### Stage 5: Tune Hyperparameters

**Goal**: Find the best configuration through systematic exploration.

**Best practices**:
1. **Use random search**, not grid search (neural nets have varying parameter sensitivity)
2. **Coarse-to-fine**: Start with wide ranges, then narrow around promising regions
3. **Use log scale** for learning rates and weight decay (e.g., 10^uniform(-4, -1))
4. **Monitor multiple metrics**: Loss, accuracy, training speed
5. **Track everything**: Use MLflow, Weights & Biases, or TensorBoard

**Key hyperparameters (in rough order of importance)**:
- Learning rate (often the most impactful)
- Learning rate schedule
- Model architecture (depth, width)
- Batch size
- Weight decay
- Dropout rates

**Warning**: Don't tune endlessly. Get 80% of gains from 20% of effort, then move on.

See [references/stage5-tuning.md](references/stage5-tuning.md) for hyperparameter search strategies.

---

### Stage 6: Squeeze Out the Juice

**Goal**: Extract final performance gains through ensemble methods and extended training.

**Final optimizations**:
1. **Ensemble models**: Train multiple models with different random seeds
2. **Average predictions**: Combine predictions from ensemble members
3. **Extend training**: Train longer once you know hyperparameters work
4. **Test-time augmentation**: Average predictions over augmented versions of test inputs
5. **Calibration**: Adjust confidence scores if needed

**Diminishing returns**: These techniques add complexity. Only apply when marginal gains are worth the engineering cost.

See [references/stage6-final.md](references/stage6-final.md) for ensemble and optimization techniques.

---

## Workflow Integration

When working on neural network training tasks:

1. **Starting new projects**: Follow stages 1-6 in order
2. **Debugging issues**: Return to the last verified stage and rebuild forward
3. **Improving existing models**: Jump to the appropriate stage based on current status
4. **Stuck on convergence**: Usually a Stage 2 or 3 issue - verify infrastructure and overfitting capability

## Common Failure Patterns and Solutions

| Symptom | Likely Stage | Solution |
|---------|--------------|----------|
| Loss is NaN | Stage 2 | Check learning rate, loss computation, initialization |
| Can't overfit single batch | Stage 2/3 | Fundamental bug - check model forward pass, loss, optimizer |
| Training loss decreases but val loss doesn't | Stage 4 | Need regularization - start with more data |
| Loss oscillates wildly | Stage 2/5 | Learning rate too high, or batch size too small |
| Training is very slow | Stage 2/5 | Check batch size, model size, data loading pipeline |

---

## Anti-Patterns to Avoid

1. **Premature optimization**: Don't tune hyperparameters before verifying you can overfit
2. **Skipping data inspection**: Jumping straight to modeling without understanding data
3. **Complex baselines**: Starting with a sophisticated architecture before trying simple ones
4. **Ignoring initialization**: Not checking if loss at init makes sense
5. **Silent failures**: Not adding enough verification and visualization
6. **Kitchen sink approach**: Adding multiple changes at once instead of isolating variables

---

## Quick Start Checklist

Before beginning any neural network training:

- [ ] Have I manually inspected hundreds/thousands of data examples?
- [ ] Have I verified my data pipeline outputs what I expect?
- [ ] Have I checked loss at initialization matches theory?
- [ ] Can I overfit a single batch of 2-8 examples?
- [ ] Have I fixed random seeds for reproducibility?
- [ ] Do I have visualization of training curves?

If any answer is "no", address it before proceeding.
