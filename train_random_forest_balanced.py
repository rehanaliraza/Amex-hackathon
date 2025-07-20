#!/usr/bin/env python3
"""
Random Forest Training with Class Imbalance Handling
Uses various techniques to handle imbalanced dataset effectively
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter
import warnings
import time
import psutil
import joblib
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

def check_system_resources():
    """Check if system has enough resources"""
    print("🖥️ SYSTEM RESOURCE CHECK")
    print("=" * 50)
    
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"💻 System Info:")
    print(f"   Total RAM: {memory_gb:.1f} GB")
    print(f"   Available RAM: {available_gb:.1f} GB")
    print(f"   Memory usage: {memory.percent:.1f}%")
    print(f"   CPU cores: {cpu_count}")
    
    return available_gb >= 2.0  # Need at least 2GB for Random Forest

def load_and_analyze_data(sample_size=None):
    """Load data and analyze class imbalance"""
    print(f"\n📊 LOADING AND ANALYZING DATA")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load training data
    print("🔄 Loading training data...")
    train_df = pd.read_parquet('train_data.parquet')
    
    # Sample if requested
    if sample_size and sample_size < len(train_df):
        print(f"📊 Sampling {sample_size:,} from {len(train_df):,} total rows")
        train_df = train_df.sample(n=sample_size, random_state=42)
    
    print(f"✅ Training data: {train_df.shape[0]:,} × {train_df.shape[1]} columns")
    print(f"⏱️ Load time: {time.time() - start_time:.2f} seconds")
    
    # Analyze class imbalance
    print(f"\n🎯 CLASS IMBALANCE ANALYSIS:")
    print("=" * 40)
    
    target_dist = train_df['y'].value_counts().sort_index()
    total_samples = len(train_df)
    
    print(f"Total samples: {total_samples:,}")
    for class_val, count in target_dist.items():
        pct = (count / total_samples) * 100
        print(f"   Class {class_val}: {count:,} ({pct:.2f}%)")
    
    # Calculate imbalance ratio
    if len(target_dist) == 2:
        minority_class = target_dist.idxmin()
        majority_class = target_dist.idxmax()
        imbalance_ratio = target_dist[majority_class] / target_dist[minority_class]
        
        print(f"\n📈 Imbalance Metrics:")
        print(f"   Minority class: {minority_class} ({target_dist[minority_class]:,} samples)")
        print(f"   Majority class: {majority_class} ({target_dist[majority_class]:,} samples)")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print(f"   ⚠️ HIGHLY IMBALANCED - Need special handling!")
        elif imbalance_ratio > 3:
            print(f"   ⚠️ MODERATELY IMBALANCED - Consider balancing techniques")
        else:
            print(f"   ✅ RELATIVELY BALANCED")
    
    return train_df, target_dist

def preprocess_features(train_df):
    """Preprocess features for Random Forest"""
    print(f"\n🔧 FEATURE PREPROCESSING")
    print("=" * 40)
    
    # Extract features and target
    feature_cols = [col for col in train_df.columns if col.startswith('f')]
    X = train_df[feature_cols].copy()
    y = pd.to_numeric(train_df['y'], errors='coerce')
    
    print(f"   Initial features: {len(feature_cols)}")
    print(f"   Initial missing values: {X.isnull().sum().sum():,}")
    
    # Convert object columns to numeric
    print("   Converting object columns to numeric...")
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Remove completely empty columns
    valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
    X_clean = X[valid_cols].copy()
    
    print(f"   Features after cleaning: {len(valid_cols)} (removed {len(feature_cols) - len(valid_cols)} empty)")
    print(f"   Missing values after cleaning: {X_clean.isnull().sum().sum():,}")
    
    # Handle missing values with median imputation
    print("   Applying median imputation...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )
    
    print(f"   After imputation: {X_imputed.isnull().sum().sum()} missing values")
    print(f"✅ Preprocessing complete")
    
    return X_imputed, y, imputer

def apply_balancing_techniques(X, y, technique='smote'):
    """Apply different balancing techniques"""
    print(f"\n⚖️ APPLYING BALANCING TECHNIQUE: {technique.upper()}")
    print("=" * 50)
    
    print(f"Original distribution: {Counter(y)}")
    
    if technique == 'smote':
        # SMOTE oversampling
        balancer = SMOTE(random_state=42, k_neighbors=5)
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        
    elif technique == 'random_oversample':
        # Random oversampling
        balancer = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        
    elif technique == 'random_undersample':
        # Random undersampling
        balancer = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        
    elif technique == 'smoteenn':
        # SMOTE + Edited Nearest Neighbours
        balancer = SMOTEENN(random_state=42)
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        
    elif technique == 'none':
        # No balancing, just return original
        X_balanced, y_balanced = X, y
        
    else:
        raise ValueError(f"Unknown balancing technique: {technique}")
    
    print(f"Balanced distribution: {Counter(y_balanced)}")
    print(f"Data shape change: {X.shape} → {X_balanced.shape}")
    
    return X_balanced, y_balanced

def train_random_forest_models(X, y, balancing_techniques=['none', 'class_weight', 'smote']):
    """Train Random Forest with different balancing approaches"""
    print(f"\n🌲 TRAINING RANDOM FOREST MODELS")
    print("=" * 60)
    
    results = {}
    
    for technique in balancing_techniques:
        print(f"\n🔄 Training with {technique.upper()} balancing...")
        start_time = time.time()
        
        # Apply balancing if needed
        if technique in ['smote', 'random_oversample', 'random_undersample', 'smoteenn']:
            X_train, y_train = apply_balancing_techniques(X, y, technique)
        else:
            X_train, y_train = X, y
        
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Configure Random Forest based on technique
        if technique == 'class_weight':
            rf_params = {
                'n_estimators': 100,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            }
        else:
            rf_params = {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            }
        
        # Train model
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train_split, y_train_split)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        auc_score = roc_auc_score(y_val, y_pred_proba)
        training_time = time.time() - start_time
        
        # Store results
        results[technique] = {
            'model': model,
            'auc_score': auc_score,
            'training_time': training_time,
            'predictions_proba': y_pred_proba,
            'predictions': y_pred,
            'y_val': y_val,
            'training_samples': len(X_train_split),
            'validation_samples': len(X_val)
        }
        
        print(f"   ✅ AUC: {auc_score:.4f}, Time: {training_time:.2f}s")
        print(f"   Training samples: {len(X_train_split):,}, Validation: {len(X_val):,}")
        
        # Clear memory
        del X_train, y_train, X_train_split, X_val, y_train_split, y_val
        gc.collect()
    
    return results

def evaluate_and_compare_models(results):
    """Evaluate and compare different models"""
    print(f"\n📊 MODEL COMPARISON")
    print("=" * 60)
    
    # Sort by AUC score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc_score'], reverse=True)
    
    print(f"{'Technique':<20} {'AUC Score':<12} {'Training Time':<15} {'Samples':<12}")
    print("-" * 65)
    
    for technique, result in sorted_results:
        print(f"{technique:<20} {result['auc_score']:<12.4f} {result['training_time']:<15.2f} {result['training_samples']:<12,}")
    
    # Best model
    best_technique, best_result = sorted_results[0]
    print(f"\n🏆 BEST MODEL: {best_technique.upper()}")
    print(f"   AUC Score: {best_result['auc_score']:.4f}")
    print(f"   Training Time: {best_result['training_time']:.2f}s")
    
    # Detailed evaluation of best model
    print(f"\n📈 DETAILED EVALUATION OF BEST MODEL")
    print("=" * 50)
    
    y_val = best_result['y_val']
    y_pred = best_result['predictions']
    y_pred_proba = best_result['predictions_proba']
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"   Predicted:    0      1")
    print(f"Actual 0:    {cm[0,0]:6} {cm[0,1]:6}")
    print(f"Actual 1:    {cm[1,0]:6} {cm[1,1]:6}")
    
    # Prediction distribution
    pred_dist = Counter(y_pred)
    total_preds = len(y_pred)
    print(f"\nPrediction Distribution:")
    for pred_val, count in sorted(pred_dist.items()):
        pct = (count / total_preds) * 100
        print(f"   Class {pred_val}: {count:,} ({pct:.2f}%)")
    
    return best_technique, best_result

def cross_validate_best_model(best_model, X, y, cv_folds=5):
    """Perform cross-validation on the best model"""
    print(f"\n🔄 CROSS-VALIDATION")
    print("=" * 40)
    
    # Stratified K-Fold for imbalanced data
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(best_model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    
    print(f"Cross-validation AUC scores:")
    for i, score in enumerate(cv_scores):
        print(f"   Fold {i+1}: {score:.4f}")
    
    print(f"\nCross-validation Summary:")
    print(f"   Mean AUC: {cv_scores.mean():.4f}")
    print(f"   Std AUC: {cv_scores.std():.4f}")
    print(f"   95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")
    
    return cv_scores

def analyze_feature_importance(model, feature_names, top_n=20):
    """Analyze feature importance"""
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"Top {top_n} Most Important Features:")
        print("-" * 40)
        for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<10} {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None

def save_model_and_results(model, preprocessor, technique, results):
    """Save the best model and results"""
    print(f"\n💾 SAVING MODEL AND RESULTS")
    print("=" * 40)
    
    # Save model
    model_filename = f'best_model_random_forest_{technique}.pkl'
    preprocessor_filename = f'data_preprocessor_random_forest_{technique}.pkl'
    results_filename = f'training_results_random_forest_{technique}.pkl'
    
    joblib.dump(model, model_filename)
    joblib.dump(preprocessor, preprocessor_filename)
    joblib.dump(results, results_filename)
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Preprocessor saved: {preprocessor_filename}")
    print(f"✅ Results saved: {results_filename}")
    
    return model_filename, preprocessor_filename, results_filename

def main():
    """Main training function"""
    print("🚀 RANDOM FOREST TRAINING WITH CLASS IMBALANCE HANDLING")
    print("=" * 70)
    
    # Check system resources
    if not check_system_resources():
        print("⚠️ Warning: Limited system resources. Consider using smaller sample.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Determine sample size based on available memory
        memory_gb = psutil.virtual_memory().available / (1024**3)
        if memory_gb >= 6:
            sample_size = 100000  # 100K samples
        elif memory_gb >= 4:
            sample_size = 77000   # 77K samples (10%)
        else:
            sample_size = 50000   # 50K samples
        
        print(f"\n🎯 TRAINING CONFIGURATION:")
        print(f"   Sample size: {sample_size:,} samples")
        print(f"   Available memory: {memory_gb:.1f} GB")
        
        # Load and analyze data
        train_df, target_dist = load_and_analyze_data(sample_size=sample_size)
        
        # Preprocess features
        X, y, imputer = preprocess_features(train_df)
        
        # Clean up original dataframe
        del train_df
        gc.collect()
        
        # Define balancing techniques to try
        balancing_techniques = ['none', 'class_weight', 'smote']
        
        # Add more techniques if we have enough memory
        if memory_gb >= 4:
            balancing_techniques.extend(['random_oversample', 'smoteenn'])
        
        print(f"\n🎯 BALANCING TECHNIQUES TO TRY:")
        for i, technique in enumerate(balancing_techniques):
            print(f"   {i+1}. {technique.upper()}")
        
        # Train models with different balancing techniques
        results = train_random_forest_models(X, y, balancing_techniques)
        
        # Evaluate and compare models
        best_technique, best_result = evaluate_and_compare_models(results)
        
        # Cross-validate best model
        best_model = best_result['model']
        cv_scores = cross_validate_best_model(best_model, X, y)
        
        # Analyze feature importance
        importance_df = analyze_feature_importance(best_model, X.columns)
        
        # Save best model
        model_files = save_model_and_results(best_model, imputer, best_technique, results)
        
        # Final summary
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 70)
        print(f"✅ Successfully completed Random Forest training with class imbalance handling")
        print(f"📊 Dataset: {len(X):,} samples, {len(X.columns)} features")
        print(f"🏆 Best technique: {best_technique.upper()}")
        print(f"📈 Best validation AUC: {best_result['auc_score']:.4f}")
        print(f"📈 Cross-validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"⏱️ Best model training time: {best_result['training_time']:.2f}s")
        print(f"💾 Model saved: {model_files[0]}")
        
        # System performance
        memory_info = psutil.virtual_memory()
        print(f"\n💻 System Performance:")
        print(f"   Final memory usage: {memory_info.percent:.1f}%")
        print(f"   Available memory: {memory_info.available / (1024**3):.1f} GB")
        
        print(f"\n🎉 RANDOM FOREST TRAINING COMPLETE!")
        print(f"   Ready to create submission with balanced Random Forest model!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 Next: Update create_submission.py to use the new Random Forest model!")
    else:
        print(f"\n⚠️ Training incomplete - check system resources and try again") 