# Memory Optimization Techniques for Large Dataset Training

## 🎯 Problem: Training on Full Dataset with Limited Memory

When you have a large dataset (770K samples, 366 features) but limited RAM (8GB), you need special techniques to train effectively.

## 🔧 Techniques Implemented

### 1. **Batch Processing (Mini-batch Training)**
```python
# Instead of loading all data at once:
# train_df = pd.read_parquet('train_data.parquet')  # 12.76 GB!

# Load in small batches:
for batch_idx in range(n_batches):
    batch_df = pd.read_parquet('train_data.parquet', 
                             skiprows=range(1, start_idx + 1),
                             nrows=batch_size)
    # Process batch...
```

**Benefits:**
- Memory usage stays constant regardless of dataset size
- Can process datasets larger than available RAM
- Allows for progress monitoring

### 2. **Incremental Learning (Online Learning)**
```python
# SGD Classifier supports incremental learning
model = SGDClassifier(loss='log_loss', class_weight='balanced')

# First batch: fit()
model.fit(X_batch_1, y_batch_1)

# Subsequent batches: partial_fit()
for batch in remaining_batches:
    model.partial_fit(X_batch, y_batch)
```

**Benefits:**
- Learns from entire dataset without loading it all
- Model improves with each batch
- Memory efficient

### 3. **Ensemble of Batch Models**
```python
# Train multiple models on different batches
models = []
for batch_idx in range(n_batches):
    batch_data = load_random_batch(batch_size)
    model = RandomForestClassifier(n_estimators=20)
    model.fit(batch_data)
    models.append(model)

# Combine predictions
ensemble_prediction = np.mean([model.predict_proba(X) for model in models], axis=0)
```

**Benefits:**
- Leverages diversity from different data subsets
- Can use complex models (Random Forest, Gradient Boosting)
- Often better than single model on subset

### 4. **Memory Management**
```python
import gc

# Explicit memory cleanup
del batch_df, X_batch, y_batch
gc.collect()

# Monitor memory usage
memory_info = psutil.virtual_memory()
print(f"Memory usage: {memory_info.percent:.1f}%")
```

**Benefits:**
- Prevents memory leaks
- Allows monitoring of resource usage
- Ensures stable training

### 5. **Feature Selection/Reduction**
```python
# Remove empty features
valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
X_clean = X[valid_cols]

# Use only most important features (from previous training)
important_features = ['f366', 'f223', 'f132', 'f363', 'f210']  # Top 5
X_reduced = X[important_features]
```

**Benefits:**
- Reduces memory footprint
- Faster training
- Often better generalization

## 📊 Comparison of Approaches

| Approach | Memory Usage | Training Time | Model Quality | Complexity |
|----------|-------------|---------------|---------------|------------|
| **10% Sample** | Low | Fast | Good | Simple |
| **Incremental (SGD)** | Very Low | Medium | Good | Medium |
| **Batch Ensemble** | Low | Medium | Very Good | Medium |
| **Feature Selection** | Low | Fast | Good | Simple |

## 🚀 Implementation Strategy

### Strategy 1: Incremental Learning (Recommended)
```bash
python train_full_dataset.py
# Choose option 1
```

**Best for:**
- Linear models (Logistic Regression, SVM)
- When you want to see ALL data
- Limited memory systems

### Strategy 2: Batch Ensemble
```bash
python train_full_dataset.py
# Choose option 2
```

**Best for:**
- Tree-based models (Random Forest, XGBoost)
- When you want model diversity
- Better prediction quality

## 💡 Additional Optimization Tips

### 1. **Data Type Optimization**
```python
# Use smaller data types
X = X.astype('float32')  # Instead of float64
y = y.astype('int8')     # Instead of int64
```

### 2. **Chunked File Reading**
```python
# Read in chunks
chunk_size = 10000
for chunk in pd.read_parquet('train_data.parquet', chunksize=chunk_size):
    process_chunk(chunk)
```

### 3. **Feature Hashing**
```python
from sklearn.feature_extraction import FeatureHasher
# Reduce feature dimensionality
hasher = FeatureHasher(n_features=1000, input_type='string')
X_hashed = hasher.transform(X)
```

### 4. **Gradient Accumulation**
```python
# Accumulate gradients over multiple batches
optimizer.zero_grad()
for batch in batches:
    loss = model(batch)
    loss.backward()
optimizer.step()
```

## 🎯 Expected Results

### Memory Usage Comparison:
- **Original approach**: 12.76 GB (won't fit)
- **10% sample**: 1.3 GB ✅
- **Batch processing**: 0.2 GB per batch ✅
- **Incremental learning**: 0.2 GB constant ✅

### Performance Expectations:
- **10% sample**: AUC ~0.93
- **Incremental (full data)**: AUC ~0.94-0.95
- **Batch ensemble**: AUC ~0.95-0.96

## 🔧 Troubleshooting

### If you get "Memory Error":
1. Reduce batch size: `batch_size=5000`
2. Close other applications
3. Use feature selection
4. Try incremental learning

### If training is slow:
1. Increase batch size (if memory allows)
2. Use fewer features
3. Use simpler models
4. Enable parallel processing

### If model quality is poor:
1. Increase number of batches
2. Use ensemble methods
3. Tune hyperparameters
4. Add feature engineering

## 📈 Next Steps

1. **Run the full dataset training**:
   ```bash
   python train_full_dataset.py
   ```

2. **Update submission script** to use new model

3. **Compare results** with 10% model

4. **Consider advanced techniques**:
   - XGBoost with early stopping
   - Neural networks with batch normalization
   - Distributed training (if multiple machines available)

This approach allows you to train on the full 770K samples even with just 8GB RAM! 