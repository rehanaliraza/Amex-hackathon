#!/usr/bin/env python3
"""
Simple Random Forest Training with Class Weight Balancing
Fast and effective approach for imbalanced datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from collections import Counter
import warnings
import time
import psutil
import joblib
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

def main():
    """Simple Random Forest training with class balancing"""
    print("🚀 SIMPLE RANDOM FOREST TRAINING WITH CLASS BALANCING")
    print("=" * 65)
    
    # Check memory
    memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"💻 Available memory: {memory_gb:.1f} GB")
    
    # Determine sample size
    if memory_gb >= 6:
        sample_size = 100000
    elif memory_gb >= 4:
        sample_size = 77000
    else:
        sample_size = 50000
    
    print(f"📊 Training on {sample_size:,} samples")
    
    # Load data
    print("\n🔄 Loading data...")
    start_time = time.time()
    train_df = pd.read_parquet('train_data.parquet')
    
    if sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=42)
    
    print(f"✅ Data loaded: {len(train_df):,} samples in {time.time() - start_time:.2f}s")
    
    # Analyze class distribution
    print("\n🎯 Class Distribution:")
    target_dist = train_df['y'].value_counts().sort_index()
    for class_val, count in target_dist.items():
        pct = (count / len(train_df)) * 100
        print(f"   Class {class_val}: {count:,} ({pct:.2f}%)")
    
    # Calculate imbalance ratio
    if len(target_dist) == 2:
        imbalance_ratio = target_dist.max() / target_dist.min()
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Preprocess features
    print("\n🔧 Preprocessing features...")
    feature_cols = [col for col in train_df.columns if col.startswith('f')]
    X = train_df[feature_cols].copy()
    y = pd.to_numeric(train_df['y'], errors='coerce')
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Missing values: {X.isnull().sum().sum():,}")
    
    # Convert to numeric and clean
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Remove empty columns
    valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
    X = X[valid_cols]
    print(f"   Valid features: {len(valid_cols)}")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Clean up
    del train_df, X
    gc.collect()
    
    # Split data
    print("\n📊 Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    
    # Train Random Forest with balanced class weights
    print("\n🌲 Training Random Forest with balanced class weights...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',  # This handles class imbalance
        n_jobs=-1,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f}s")
    
    # Evaluate
    print("\n📊 Evaluating model...")
    y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
    y_pred = rf_model.predict(X_val)
    
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"   Validation AUC: {auc_score:.4f}")
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_val, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   Predicted:    0      1")
    print(f"Actual 0:    {cm[0,0]:6} {cm[0,1]:6}")
    print(f"Actual 1:    {cm[1,0]:6} {cm[1,1]:6}")
    
    # Prediction distribution
    pred_dist = Counter(y_pred)
    print(f"\n📊 Prediction Distribution:")
    for pred_val, count in sorted(pred_dist.items()):
        pct = (count / len(y_pred)) * 100
        print(f"   Class {pred_val}: {count:,} ({pct:.2f}%)")
    
    # Cross-validation
    print("\n🔄 Cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_imputed, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    
    print(f"   CV AUC scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"   Mean CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    print("\n🔍 Top 10 Feature Importances:")
    importances = rf_model.feature_importances_
    feature_importance = list(zip(X_imputed.columns, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"   {i+1:2d}. {feature:<12} {importance:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    model_filename = 'best_model_simple_rf_balanced.pkl'
    preprocessor_filename = 'data_preprocessor_simple_rf.pkl'
    
    joblib.dump(rf_model, model_filename)
    joblib.dump(imputer, preprocessor_filename)
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Preprocessor saved: {preprocessor_filename}")
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 50)
    print(f"✅ Random Forest training complete")
    print(f"📊 Dataset: {len(X_imputed):,} samples, {len(X_imputed.columns)} features")
    print(f"📈 Validation AUC: {auc_score:.4f}")
    print(f"📈 Cross-validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"⏱️ Training time: {training_time:.2f}s")
    print(f"💾 Model: {model_filename}")
    
    # Memory usage
    memory_info = psutil.virtual_memory()
    print(f"\n💻 Final memory usage: {memory_info.percent:.1f}%")
    
    print(f"\n🎉 SIMPLE RANDOM FOREST TRAINING COMPLETE!")
    print(f"   Ready to create submission!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n🚀 Next: Use create_submission.py with the new Random Forest model!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc() 