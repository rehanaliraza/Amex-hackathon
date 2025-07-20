#!/usr/bin/env python3
"""
Simple Full Dataset Training with Memory Optimization
Uses incremental learning with chunked data processing
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
import time
import psutil
import joblib
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

def check_memory():
    """Check current memory usage"""
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}% ({memory.available / (1024**3):.1f} GB available)")
    return memory.available / (1024**3)

def train_incremental_full_dataset():
    """Train on full dataset using incremental learning"""
    print("🚀 SIMPLE FULL DATASET TRAINING")
    print("=" * 60)
    
    # Check initial memory
    available_memory = check_memory()
    if available_memory < 1.0:
        print("⚠️ Warning: Less than 1GB available. Close other applications.")
        return False
    
    # Load and analyze full dataset
    print("\n📊 Loading full dataset...")
    start_time = time.time()
    
    try:
        train_df = pd.read_parquet('train_data.parquet')
        print(f"✅ Dataset loaded: {len(train_df):,} samples, {len(train_df.columns)} columns")
        print(f"⏱️ Load time: {time.time() - start_time:.2f} seconds")
        
        # Check memory after loading
        check_memory()
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Analyze target distribution
    print(f"\n🎯 Target Analysis:")
    target_counts = train_df['y'].value_counts()
    for val, count in target_counts.items():
        pct = (count / len(train_df)) * 100
        print(f"   Class {val}: {count:,} ({pct:.1f}%)")
    
    # Prepare features
    print(f"\n🔧 Feature Preparation:")
    feature_cols = [col for col in train_df.columns if col.startswith('f')]
    print(f"   Initial features: {len(feature_cols)}")
    
    # Convert to numeric and clean
    X = train_df[feature_cols].copy()
    y = pd.to_numeric(train_df['y'], errors='coerce')
    
    # Convert object columns to numeric
    print("   Converting object columns to numeric...")
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Remove completely empty columns
    valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
    X = X[valid_cols]
    print(f"   Features after cleaning: {len(valid_cols)}")
    
    # Check memory after preprocessing
    check_memory()
    
    # Split into train/validation (80/20)
    print(f"\n📊 Creating train/validation split...")
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    
    # Handle missing values
    print(f"\n🔧 Handling missing values...")
    print(f"   Missing values before imputation: {X_train.isnull().sum().sum():,}")
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    print(f"   Missing values after imputation: {X_train_imputed.isnull().sum().sum()}")
    
    # Clean up original data
    del train_df, X, X_train, X_val
    gc.collect()
    check_memory()
    
    # Train model with incremental learning
    print(f"\n🤖 Training SGD Classifier...")
    
    model = SGDClassifier(
        loss='log_loss',  # For probability estimates
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
        learning_rate='adaptive',
        eta0=0.01,
        verbose=1
    )
    
    # Train in chunks to manage memory
    chunk_size = 50000
    n_chunks = (len(X_train_imputed) + chunk_size - 1) // chunk_size
    
    print(f"   Training in {n_chunks} chunks of {chunk_size:,} samples each...")
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(X_train_imputed))
        
        X_chunk = X_train_imputed.iloc[start_idx:end_idx]
        y_chunk = y_train.iloc[start_idx:end_idx]
        
        # Remove any remaining NaN values
        valid_mask = ~(X_chunk.isnull().any(axis=1) | y_chunk.isnull())
        X_chunk = X_chunk[valid_mask]
        y_chunk = y_chunk[valid_mask]
        
        if len(X_chunk) > 0:
            if chunk_idx == 0:
                model.fit(X_chunk, y_chunk)
            else:
                model.partial_fit(X_chunk, y_chunk)
            
            print(f"   ✅ Chunk {chunk_idx + 1}/{n_chunks}: {len(X_chunk):,} samples processed")
        
        # Clear chunk memory
        del X_chunk, y_chunk
        gc.collect()
        
        if (chunk_idx + 1) % 5 == 0:
            check_memory()
    
    print(f"✅ Training complete!")
    
    # Validate model
    print(f"\n📊 Validating model...")
    
    # Validate in chunks to manage memory
    val_predictions = []
    val_targets = []
    
    val_chunk_size = 25000
    n_val_chunks = (len(X_val_imputed) + val_chunk_size - 1) // val_chunk_size
    
    for chunk_idx in range(n_val_chunks):
        start_idx = chunk_idx * val_chunk_size
        end_idx = min(start_idx + val_chunk_size, len(X_val_imputed))
        
        X_val_chunk = X_val_imputed.iloc[start_idx:end_idx]
        y_val_chunk = y_val.iloc[start_idx:end_idx]
        
        # Remove NaN values
        valid_mask = ~(X_val_chunk.isnull().any(axis=1) | y_val_chunk.isnull())
        X_val_chunk = X_val_chunk[valid_mask]
        y_val_chunk = y_val_chunk[valid_mask]
        
        if len(X_val_chunk) > 0:
            y_pred_proba = model.predict_proba(X_val_chunk)[:, 1]
            val_predictions.extend(y_pred_proba)
            val_targets.extend(y_val_chunk)
        
        # Clear memory
        del X_val_chunk, y_val_chunk
        gc.collect()
    
    # Calculate validation AUC
    if len(val_predictions) > 0:
        val_auc = roc_auc_score(val_targets, val_predictions)
        print(f"✅ Validation AUC: {val_auc:.4f}")
        print(f"   Validation samples: {len(val_predictions):,}")
    else:
        print("❌ No validation samples available")
        val_auc = None
    
    # Clean up validation data
    del X_train_imputed, X_val_imputed, y_train, y_val, val_predictions, val_targets
    gc.collect()
    
    # Save model
    print(f"\n💾 Saving model...")
    model_filename = 'best_model_full_simple_sgd.pkl'
    preprocessor_filename = 'data_preprocessor_full_simple.pkl'
    
    joblib.dump(model, model_filename)
    joblib.dump(imputer, preprocessor_filename)
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Preprocessor saved: {preprocessor_filename}")
    
    # Final summary
    print(f"\n🎯 TRAINING SUMMARY:")
    print(f"   Training samples: {split_idx:,}")
    print(f"   Validation samples: {len(train_df) - split_idx:,}")
    print(f"   Features used: {len(valid_cols)}")
    print(f"   Validation AUC: {val_auc:.4f}" if val_auc else "   No validation AUC")
    print(f"   Model type: SGD Classifier")
    
    check_memory()
    
    return True

def main():
    """Main function"""
    try:
        success = train_incremental_full_dataset()
        
        if success:
            print(f"\n🎉 FULL DATASET TRAINING SUCCESSFUL!")
            print(f"   Ready to create submission with full dataset model!")
        else:
            print(f"\n❌ Training failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 Next: Update create_submission.py to use the new full dataset model!")
    else:
        print(f"\n⚠️ Training incomplete - check memory and try again") 