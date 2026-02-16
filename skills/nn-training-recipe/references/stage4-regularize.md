# Stage 4: Regularization

## Goal

Improve validation performance by adding constraints that help the model generalize better. You've proven the model can fit training data (Stage 3), now make it work on unseen data.

## Priority Order: Data First, Engineering Second

**Critical principle**: Real data is by far the most effective regularization.

### The Regularization Hierarchy

1. **More real training data** (10-100x more effective than anything else)
2. **Data augmentation** (synthetic data that preserves true invariances)
3. **Pretraining/Transfer learning** (leverage existing models)
4. **Simplify inputs** (remove noise, reduce dimensionality)
5. **Architectural constraints** (smaller models, appropriate inductive biases)
6. **Explicit regularization** (dropout, weight decay, early stopping)

Work down this list in order. Don't tune dropout rates before trying to get more data.

---

## 1. Get More Real Data

**The single most effective approach**:
- Collect more examples
- Re-label ambiguous examples more carefully
- Include edge cases and rare examples
- Balance class distributions

**Even small improvements in data quality/quantity can match weeks of engineering**.

**If you can't get more data**: Focus on the next items in the hierarchy.

---

## 2. Data Augmentation

Create synthetic training examples that preserve the true invariances of your task.

### Image Augmentation

**Standard augmentations**:
```python
import torchvision.transforms as transforms

# Basic augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

# Stronger augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# Modern augmentation (for classification)
from torchvision.transforms import AutoAugment, RandAugment

train_transform = transforms.Compose([
    AutoAugment(),  # Or RandAugment()
    transforms.ToTensor(),
])
```

**Task-specific considerations**:
- **Natural images**: Flips, crops, color jitter usually safe
- **Medical images**: Be careful - random flips might change diagnosis
- **OCR/Text detection**: Rotation and perspective transforms
- **Satellite imagery**: Rotation OK, flips OK, but preserve spatial relationships

**Warning**: Only augment in ways that preserve the label:
- ❌ Don't flip images of text (changes meaning)
- ❌ Don't color jitter grayscale medical images
- ✅ Do flip natural objects (bird species doesn't change with flip)

### Text Augmentation

```python
# Back-translation
# Translate to another language and back
"The cat sat on the mat"
  -> (to French) "Le chat était assis sur le tapis"
  -> (back to English) "The cat was sitting on the mat"

# Synonym replacement
"The quick brown fox" -> "The fast brown fox"

# Random insertion/deletion
"I love programming" -> "I really love programming"

# EDA (Easy Data Augmentation)
import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='wordnet')
augmented = aug.augment("The quick brown fox")
```

### Tabular Data Augmentation

```python
# Add small Gaussian noise
X_augmented = X + np.random.normal(0, 0.01, X.shape)

# SMOTE for imbalanced classification
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Mixup (interpolate between examples)
alpha = 0.2
lam = np.random.beta(alpha, alpha)
X_mixed = lam * X[i] + (1 - lam) * X[j]
y_mixed = lam * y[i] + (1 - lam) * y[j]
```

**Progressive augmentation strategy**:
1. Start with no augmentation (Stage 2-3)
2. Add mild augmentation first
3. Increase strength if validation improves
4. Monitor training time (heavy augmentation slows training)

---

## 3. Pretraining and Transfer Learning

Use models pretrained on large datasets:

### Vision Models

```python
import torchvision.models as models

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Option 1: Fine-tune entire model
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(2048, num_classes)  # Replace final layer

# Option 2: Freeze backbone, train only head (faster, less data needed)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)
for param in model.fc.parameters():
    param.requires_grad = True

# Option 3: Progressive unfreezing (best generalization)
# Start with frozen backbone, gradually unfreeze layers
```

### NLP Models

```python
from transformers import BertModel, BertTokenizer

# Load pretrained BERT
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add classification head
class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)
```

**When to use pretraining**:
- ✅ Your task is similar to pretraining task (e.g., ImageNet for general images)
- ✅ You have limited data (<10K examples)
- ⚠️ Be careful if your domain differs significantly (medical images, satellite)

---

## 4. Simplify Inputs

Sometimes less is more:

### Dimensionality Reduction

```python
# PCA for tabular data
from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # Reduce from 1000 features to 50
X_reduced = pca.fit_transform(X)

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)
```

### Input Resolution

```python
# Reduce image resolution
train_transform = transforms.Compose([
    transforms.Resize(64),  # Instead of 224
    transforms.ToTensor(),
])
```

Lower resolution = fewer parameters = better generalization (when data is limited).

---

## 5. Architectural Regularization

### Reduce Model Size

```python
# Was: Large model that overfits
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, 3)  # Many channels
        self.conv2 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

# Better: Smaller model
class SmallerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)   # Fewer channels
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
```

### Batch Normalization

```python
# Add batch norm (helps with generalization and training stability)
class ModelWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)  # Add batch norm
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        # ...

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # ...
```

---

## 6. Explicit Regularization Techniques

### Dropout

```python
class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)  # Drop 50% of neurons
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)  # Drop 30% of neurons
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

**Dropout rates**:
- Start with 0.5 for fully connected layers
- Use 0.2-0.3 for convolutional layers
- Don't use dropout right before the output layer

### Weight Decay (L2 Regularization)

```python
# Add L2 penalty to weights
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4  # L2 regularization strength
)

# Try different strengths: 1e-5, 1e-4, 1e-3
```

### Early Stopping

```python
# Stop training when validation loss stops improving
best_val_loss = float('inf')
patience = 10  # Number of epochs to wait
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pt'))
```

### Label Smoothing

```python
# Instead of hard targets [0, 1, 0], use soft targets [0.05, 0.9, 0.05]
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### Mixup Training

```python
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# In training loop
mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
output = model(mixed_x)
loss = mixup_criterion(criterion, output, y_a, y_b, lam)
```

---

## Iterative Regularization Strategy

**Don't add everything at once**. Add techniques one at a time and measure impact:

1. **Baseline**: Train overfit model (Stage 3 result)
   - Record validation performance

2. **Add augmentation**: Implement data augmentation
   - Train and record validation performance
   - Keep if improvement > 1-2%

3. **Try pretraining**: If applicable
   - Compare to augmentation-only
   - Keep if improvement > 2-3%

4. **Add dropout**: Start with 0.3-0.5
   - Monitor both train and val loss
   - Keep if val improves without train degrading too much

5. **Add weight decay**: Try 1e-4
   - Check if combined with dropout helps

6. **Early stopping**: Set patience based on how long training takes
   - Always keep this for final model

### Monitoring Regularization

```python
# Track train/val gap to guide regularization
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

# Compute overfitting gap
gap = train_acc - val_acc
print(f"Train-Val Gap: {gap:.3f}")

# If gap > 15-20%, more regularization needed
# If gap < 5%, might be underfitting
```

---

## Success Criteria for Stage 4

You've successfully completed regularization when:

- [ ] Validation performance is close to training performance (gap < 10-15%)
- [ ] Validation performance has plateaued despite trying multiple regularization techniques
- [ ] You've exhausted reasonable data augmentation strategies
- [ ] Model is no longer egregiously overfitting

**Note**: Some train-val gap is normal and expected. Perfect elimination of the gap may indicate underfitting.

---

## Common Mistakes

1. **Over-regularization**: Adding too many constraints makes the model underfit
   - Symptom: Training accuracy drops significantly
   - Solution: Remove some regularization

2. **Under-regularization**: Not adding enough constraints
   - Symptom: Large train-val gap persists
   - Solution: Add stronger augmentation or dropout

3. **Wrong augmentation**: Using augmentation that changes labels
   - Symptom: Training accuracy decreases
   - Solution: Review augmentation for label preservation

4. **Ignoring data quality**: Focusing on engineering before data
   - Solution: Always try to get more/better data first

---

## Next Steps

Once validation performance has plateaued with regularization, move to Stage 5 (Hyperparameter Tuning) to squeeze out the final performance gains through systematic architecture and hyperparameter search.
