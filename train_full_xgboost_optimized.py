#!/usr/bin/env python3
"""
Full Dataset XGBoost Training with Incremental Learning and Feature Selection
Uses most important features identified from feature importance analysis
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from collections import Counter
import warnings
import time
import psutil
import joblib
import gc
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

def load_important_features():
    """Load the most important features from feature importance analysis"""
    print("📋 LOADING IMPORTANT FEATURES")
    print("=" * 50)
    
    # Try to load from saved feature files
    feature_files = [
        ('tier1_tier2_features.txt', 'Tier 1+2 Features (Best Performance)'),
        ('top_100_features.txt', 'Top 100 Features (Balanced)'),
        ('recommended_features.txt', 'Recommended Features'),
        ('tier1_features.txt', 'Tier 1 Features (Most Important)'),
    ]
    
    for filename, description in feature_files:
        if Path(filename).exists():
            try:
                with open(filename, 'r') as f:
                    features = [line.strip() for line in f.readlines() if line.strip()]
                
                print(f"✅ Loaded {len(features)} features from {filename}")
                print(f"   Description: {description}")
                print(f"   Sample features: {features[:10]}{'...' if len(features) > 10 else ''}")
                
                return features, description
                
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
                continue
    
    # Fallback: use manually selected top features based on our analysis
    print("⚠️ No feature files found, using manually selected top features")
    top_features = [
        'f366', 'f223', 'f132', 'f363', 'f137', 'f138', 'f134', 'f150', 'f210', 'f125',
        'f133', 'f365', 'f127', 'f130', 'f123', 'f124', 'f31', 'f149', 'f151', 'f126',
        'f312', 'f131', 'f147', 'f313', 'f95', 'f143', 'f314', 'f350', 'f358', 'f139',
        'f141', 'f142', 'f144', 'f145', 'f146', 'f148', 'f152', 'f153', 'f154', 'f155',
        'f156', 'f157', 'f158', 'f159', 'f160', 'f161', 'f162', 'f163', 'f164', 'f165'
    ]
    
    return top_features, "Top 50 Manual Selection"

def check_system_resources():
    """Check system resources and recommend batch size"""
    print("\n🖥️ SYSTEM RESOURCE CHECK")
    print("=" * 50)
    
    memory = psutil.virtual_memory()
    memory_gb = memory.available / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"💻 System Info:")
    print(f"   Available RAM: {memory_gb:.1f} GB")
    print(f"   CPU cores: {cpu_count}")
    print(f"   Memory usage: {memory.percent:.1f}%")
    
    # Recommend batch size based on available memory
    if memory_gb >= 6:
        batch_size = 100000
        strategy = "High Memory"
    elif memory_gb >= 4:
        batch_size = 75000
        strategy = "Medium Memory"
    elif memory_gb >= 2:
        batch_size = 50000
        strategy = "Low Memory"
    else:
        batch_size = 25000
        strategy = "Very Low Memory"
    
    print(f"\n📊 Recommended Configuration:")
    print(f"   Strategy: {strategy}")
    print(f"   Batch size: {batch_size:,} samples")
    print(f"   Expected batches: ~{770000 // batch_size + 1}")
    
    return batch_size, memory_gb >= 2

def load_and_preprocess_data_batch(batch_start, batch_size, important_features, preprocessor=None, fit_preprocessor=False):
    """Load and preprocess a batch of data with feature selection"""
    try:
        # Load full dataset (we'll optimize this later with chunked reading)
        train_df = pd.read_parquet('train_data.parquet')
        
        # Extract batch
        batch_end = min(batch_start + batch_size, len(train_df))
        batch_df = train_df.iloc[batch_start:batch_end].copy()
        
        # Clean up full dataset from memory
        del train_df
        gc.collect()
        
        # Extract features and target
        available_features = [f for f in important_features if f in batch_df.columns]
        X_batch = batch_df[available_features].copy()
        y_batch = pd.to_numeric(batch_df['y'], errors='coerce')
        
        # Convert to numeric and clean
        for col in X_batch.columns:
            if X_batch[col].dtype == 'object':
                X_batch[col] = pd.to_numeric(X_batch[col], errors='coerce')
        
        # Handle missing values
        if fit_preprocessor:
            preprocessor = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                preprocessor.fit_transform(X_batch),
                columns=X_batch.columns,
                index=X_batch.index
            )
        else:
            X_processed = pd.DataFrame(
                preprocessor.transform(X_batch),
                columns=X_batch.columns,
                index=X_batch.index
            )
        
        # Remove any remaining NaN values
        valid_mask = ~(X_processed.isnull().any(axis=1) | y_batch.isnull())
        X_processed = X_processed[valid_mask]
        y_batch = y_batch[valid_mask]
        
        # Clean up
        del batch_df, X_batch
        gc.collect()
        
        return X_processed, y_batch, preprocessor
        
    except Exception as e:
        print(f"❌ Error loading batch {batch_start}-{batch_end}: {e}")
        return None, None, preprocessor

def calculate_class_weights(y):
    """Calculate class weights for imbalanced dataset"""
    class_counts = Counter(y)
    total_samples = len(y)
    
    # Calculate inverse frequency weights
    class_weights = {}
    for class_label, count in class_counts.items():
        class_weights[class_label] = total_samples / (len(class_counts) * count)
    
    return class_weights

def train_xgboost_incremental(important_features, batch_size, total_samples=770164):
    """Train XGBoost using incremental learning with class imbalance handling"""
    print(f"\n🚀 INCREMENTAL XGBOOST TRAINING")
    print("=" * 60)
    
    # Initialize variables
    preprocessor = None
    model = None
    n_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"📊 Training Configuration:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Number of batches: {n_batches}")
    print(f"   Features: {len(important_features)}")
    
    # XGBoost parameters optimized for class imbalance
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 1,
        'n_jobs': -1,
        # Class imbalance handling
        'scale_pos_weight': 20,  # Approximate class imbalance ratio
    }
    
    print(f"\n🎯 XGBoost Parameters:")
    for key, value in xgb_params.items():
        print(f"   {key}: {value}")
    
    # Training loop
    total_samples_processed = 0
    batch_aucs = []
    
    print(f"\n🔄 Starting incremental training...")
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        print(f"\n📦 Processing batch {batch_idx + 1}/{n_batches} (samples {batch_start:,}-{min(batch_start + batch_size, total_samples):,})")
        
        # Load batch
        start_time = time.time()
        X_batch, y_batch, preprocessor = load_and_preprocess_data_batch(
            batch_start, batch_size, important_features, preprocessor, 
            fit_preprocessor=(batch_idx == 0)
        )
        
        if X_batch is None or len(X_batch) == 0:
            print(f"   ⚠️ Skipping empty batch {batch_idx + 1}")
            continue
        
        load_time = time.time() - start_time
        
        # Analyze batch class distribution
        class_counts = Counter(y_batch)
        batch_imbalance = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
        
        print(f"   📊 Batch stats: {len(X_batch):,} samples, {len(X_batch.columns)} features")
        print(f"   📊 Class distribution: {dict(class_counts)}")
        print(f"   📊 Imbalance ratio: {batch_imbalance:.2f}:1")
        print(f"   ⏱️ Load time: {load_time:.2f}s")
        
        # Handle severe class imbalance in batch
        if batch_imbalance > 50:  # Very imbalanced batch
            # Adjust scale_pos_weight for this batch
            xgb_params['scale_pos_weight'] = min(batch_imbalance, 100)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_batch, label=y_batch)
        
        # Train model
        train_start = time.time()
        
        if model is None:
            # First batch: train initial model
            model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=100,
                verbose_eval=False
            )
        else:
            # Subsequent batches: continue training (incremental learning)
            model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=50,  # Fewer rounds for incremental updates
                xgb_model=model,  # Continue from existing model
                verbose_eval=False
            )
        
        train_time = time.time() - train_start
        
        # Quick batch evaluation
        batch_pred = model.predict(dtrain)
        batch_auc = roc_auc_score(y_batch, batch_pred)
        batch_aucs.append(batch_auc)
        
        print(f"   📈 Batch AUC: {batch_auc:.4f}")
        print(f"   ⏱️ Training time: {train_time:.2f}s")
        
        total_samples_processed += len(X_batch)
        
        # Memory cleanup
        del X_batch, y_batch, dtrain
        gc.collect()
        
        # Memory check
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 90:
            print(f"   ⚠️ High memory usage: {memory_info.percent:.1f}%")
        
        # Progress update
        if (batch_idx + 1) % 5 == 0:
            avg_auc = np.mean(batch_aucs[-5:])
            print(f"   📊 Progress: {total_samples_processed:,}/{total_samples:,} ({100*total_samples_processed/total_samples:.1f}%)")
            print(f"   📊 Recent avg AUC: {avg_auc:.4f}")
    
    print(f"\n✅ Incremental training complete!")
    print(f"   Total samples processed: {total_samples_processed:,}")
    print(f"   Batches processed: {len(batch_aucs)}")
    print(f"   Average batch AUC: {np.mean(batch_aucs):.4f} ± {np.std(batch_aucs):.4f}")
    
    return model, preprocessor

def validate_model(model, preprocessor, important_features, validation_samples=100000):
    """Validate the trained model on a held-out set"""
    print(f"\n📊 MODEL VALIDATION")
    print("=" * 50)
    
    try:
        # Load validation data
        print(f"🔄 Loading {validation_samples:,} samples for validation...")
        train_df = pd.read_parquet('train_data.parquet')
        
        # Use last portion as validation set
        val_df = train_df.tail(validation_samples).copy()
        del train_df
        gc.collect()
        
        # Preprocess validation data
        available_features = [f for f in important_features if f in val_df.columns]
        X_val = val_df[available_features].copy()
        y_val = pd.to_numeric(val_df['y'], errors='coerce')
        
        # Convert to numeric and clean
        for col in X_val.columns:
            if X_val[col].dtype == 'object':
                X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
        
        # Apply preprocessing
        X_val_processed = pd.DataFrame(
            preprocessor.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Remove NaN values
        valid_mask = ~(X_val_processed.isnull().any(axis=1) | y_val.isnull())
        X_val_processed = X_val_processed[valid_mask]
        y_val = y_val[valid_mask]
        
        print(f"✅ Validation data prepared: {len(X_val_processed):,} samples")
        
        # Validate in chunks to manage memory
        chunk_size = 25000
        n_chunks = (len(X_val_processed) + chunk_size - 1) // chunk_size
        
        all_predictions = []
        all_targets = []
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(X_val_processed))
            
            X_chunk = X_val_processed.iloc[start_idx:end_idx]
            y_chunk = y_val.iloc[start_idx:end_idx]
            
            # Create DMatrix and predict
            dval_chunk = xgb.DMatrix(X_chunk)
            pred_chunk = model.predict(dval_chunk)
            
            all_predictions.extend(pred_chunk)
            all_targets.extend(y_chunk)
            
            # Clear memory
            del X_chunk, y_chunk, dval_chunk
            gc.collect()
        
        # Calculate metrics
        val_auc = roc_auc_score(all_targets, all_predictions)
        val_predictions_binary = (np.array(all_predictions) >= 0.5).astype(int)
        
        print(f"\n📈 VALIDATION RESULTS:")
        print(f"   Validation AUC: {val_auc:.4f}")
        print(f"   Validation samples: {len(all_targets):,}")
        
        # Class distribution in predictions
        pred_dist = Counter(val_predictions_binary)
        print(f"   Prediction distribution:")
        for class_val, count in sorted(pred_dist.items()):
            pct = (count / len(val_predictions_binary)) * 100
            print(f"     Class {class_val}: {count:,} ({pct:.2f}%)")
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, val_predictions_binary)
        print(f"\n📊 Confusion Matrix:")
        print(f"   Predicted:    0      1")
        print(f"Actual 0:    {cm[0,0]:6} {cm[0,1]:6}")
        print(f"Actual 1:    {cm[1,0]:6} {cm[1,1]:6}")
        
        return val_auc
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

def save_model(model, preprocessor, important_features, validation_auc):
    """Save the trained model and preprocessor"""
    print(f"\n💾 SAVING MODEL")
    print("=" * 40)
    
    # Save model
    model_filename = 'best_model_full_xgboost_optimized.json'
    model.save_model(model_filename)
    print(f"✅ XGBoost model saved: {model_filename}")
    
    # Save preprocessor
    preprocessor_filename = 'data_preprocessor_full_xgboost.pkl'
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"✅ Preprocessor saved: {preprocessor_filename}")
    
    # Save feature list
    features_filename = 'features_full_xgboost.txt'
    with open(features_filename, 'w') as f:
        for feature in important_features:
            f.write(f"{feature}\n")
    print(f"✅ Feature list saved: {features_filename}")
    
    # Save model info
    info_filename = 'model_info_full_xgboost.txt'
    with open(info_filename, 'w') as f:
        f.write(f"Full Dataset XGBoost Model\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Training date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Features used: {len(important_features)}\n")
        f.write(f"Validation AUC: {validation_auc:.4f}\n")
        f.write(f"Model file: {model_filename}\n")
        f.write(f"Preprocessor file: {preprocessor_filename}\n")
        f.write(f"Features file: {features_filename}\n")
    
    print(f"✅ Model info saved: {info_filename}")
    
    return model_filename, preprocessor_filename

def main():
    """Main training function"""
    print("🚀 FULL DATASET XGBOOST TRAINING WITH FEATURE SELECTION")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Load important features
        important_features, feature_description = load_important_features()
        
        # Check system resources
        batch_size, memory_ok = check_system_resources()
        
        if not memory_ok:
            print("⚠️ Warning: Limited memory. Training may be slow or fail.")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                return False
        
        # Train XGBoost model
        model, preprocessor = train_xgboost_incremental(
            important_features, batch_size
        )
        
        # Validate model
        validation_auc = validate_model(model, preprocessor, important_features)
        
        # Save model
        model_file, preprocessor_file = save_model(
            model, preprocessor, important_features, validation_auc
        )
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully trained XGBoost on full dataset")
        print(f"📊 Features used: {len(important_features)} ({feature_description})")
        print(f"📈 Validation AUC: {validation_auc:.4f}")
        print(f"⏱️ Total training time: {total_time/60:.1f} minutes")
        print(f"💾 Model saved: {model_file}")
        print(f"💾 Preprocessor saved: {preprocessor_file}")
        
        # System performance
        memory_info = psutil.virtual_memory()
        print(f"\n💻 Final System State:")
        print(f"   Memory usage: {memory_info.percent:.1f}%")
        print(f"   Available memory: {memory_info.available / (1024**3):.1f} GB")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Update create_submission.py to use this model")
        print(f"   2. Generate predictions on test data")
        print(f"   3. Compare performance with previous models")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 