# karpathy-as-a-skill
converting certain aspects of how andrej karpathy operates and thinks nn training into a skill to empower your journey of deep learning with llms

# Karpathy NN Training Recipe (Concise)

A practical, verification-first workflow for training neural networks reliably.

## Core Principle
Neural networks can fail silently. Build complexity gradually and verify each step before moving on.

## 6-Stage Workflow
1. **Know your data**: inspect many samples, distributions, and labeling/data-quality issues.
2. **Build a minimal end-to-end pipeline**: fix seeds, disable augmentation first, verify inputs and initial loss, and overfit a tiny batch (2–8 samples).
3. **Overfit on purpose**: prove the model can drive training loss near zero.
4. **Regularize for validation performance**: prioritize more real data, then augmentation/pretraining, then model-size and regularization tricks.
5. **Tune hyperparameters systematically**: random search, coarse-to-fine ranges, log-scale LR/weight decay, track runs.
6. **Final optimization**: ensembles, longer training, test-time augmentation, calibration (only if worth the extra complexity).

## Fast Debug Rules
- **Can’t overfit a tiny batch** → fundamental pipeline/model bug.
- **Train improves but validation doesn’t** → regularization/data generalization issue.
- **NaN or unstable loss** → check learning rate, initialization, and loss implementation.

## Pre-Training Checklist
- [ ] Inspected a large sample of data
- [ ] Verified pipeline outputs visually
- [ ] Checked loss at initialization
- [ ] Confirmed tiny-batch overfitting works
- [ ] Fixed random seeds
- [ ] Enabled training/validation curve tracking
