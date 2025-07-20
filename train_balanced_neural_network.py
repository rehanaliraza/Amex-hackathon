#!/usr/bin/env python3
"""
Balanced Neural Network Training with Class Imbalance Handling
Uses top 100 features and multiple balancing techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
    """Check system resources"""
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
    
    return available_gb

def load_top_100_features():
    """Load the top 100 features from feature importance analysis"""
    try:
        df = pd.read_csv('feature_importance_analysis.csv')
        top_100_features = df.head(100)['feature'].tolist()
        print(f"✅ Loaded {len(top_100_features)} top features from analysis")
        return top_100_features
    except FileNotFoundError:
        # Fallback to basic features if file not found
        basic_features = ['f366', 'f223', 'f132', 'f363', 'f137', 'f138', 'f134', 'f150', 'f210', 'f125']
        print(f"⚠️ Using fallback {len(basic_features)} features")
        return basic_features

def analyze_class_imbalance(df):
    """Analyze class imbalance in the dataset"""
    print("\n🎯 CLASS IMBALANCE ANALYSIS")
    print("=" * 50)
    
    target_dist = df['y'].value_counts().sort_index()
    total_samples = len(df)
    
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
    
    return target_dist, imbalance_ratio

def preprocess_features(df, important_features):
    """Preprocess features for neural network"""
    print(f"\n🔧 FEATURE PREPROCESSING")
    print("=" * 40)
    
    # Extract available features
    available_features = [f for f in important_features if f in df.columns]
    X = df[available_features].copy()
    y = pd.to_numeric(df['y'], errors='coerce')
    
    print(f"   Requested features: {len(important_features)}")
    print(f"   Available features: {len(available_features)}")
    print(f"   Missing features: {len(important_features) - len(available_features)}")
    print(f"   Initial missing values: {X.isnull().sum().sum():,}")
    
    # Convert object columns to numeric
    print("   Converting object columns to numeric...")
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Remove completely empty columns
    valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
    X_clean = X[valid_cols].copy()
    
    print(f"   Features after cleaning: {len(valid_cols)} (removed {len(available_features) - len(valid_cols)} empty)")
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

def apply_balancing_technique(X, y, technique='smote'):
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

def train_neural_network_models(X, y, balancing_techniques=['none', 'smote', 'random_oversample']):
    """Train neural networks with different balancing approaches"""
    print(f"\n🧠 TRAINING NEURAL NETWORK MODELS")
    print("=" * 60)
    
    results = {}
    
    for technique in balancing_techniques:
        print(f"\n🔄 Training with {technique.upper()} balancing...")
        start_time = time.time()
        
        # Apply balancing if needed
        if technique in ['smote', 'random_oversample', 'random_undersample', 'smoteenn']:
            X_train, y_train = apply_balancing_technique(X, y, technique)
        else:
            X_train, y_train = X, y
        
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val)
        
        # Configure Neural Network
        nn_params = {
            'hidden_layer_sizes': (256, 128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'batch_size': 32,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'verbose': False
        }
        
        # Note: MLPClassifier doesn't support class_weight, so we rely on balancing techniques
        
        # Train model
        model = MLPClassifier(**nn_params)
        model.fit(X_train_scaled, y_train_split)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        y_pred = model.predict(X_val_scaled)
        
        auc_score = roc_auc_score(y_val, y_pred_proba)
        training_time = time.time() - start_time
        
        print(f"   ✅ Training completed in {training_time:.2f}s")
        print(f"   📊 Validation AUC: {auc_score:.4f}")
        
        # Store results
        results[technique] = {
            'model': model,
            'scaler': scaler,
            'auc_score': auc_score,
            'training_time': training_time,
            'training_samples': len(X_train),
            'y_val': y_val,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba
        }
        
        # Memory cleanup
        del X_train_scaled, X_val_scaled, y_train_split, y_val
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

def save_best_model(best_result, imputer, important_features):
    """Save the best model and preprocessors"""
    print(f"\n💾 SAVING BEST MODEL")
    print("=" * 40)
    
    model = best_result['model']
    scaler = best_result['scaler']
    
    # Save model and preprocessors
    model_filename = 'best_model_balanced_nn.pkl'
    scaler_filename = 'scaler_balanced_nn.pkl'
    imputer_filename = 'imputer_balanced_nn.pkl'
    features_filename = 'features_balanced_nn.txt'
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(imputer, imputer_filename)
    
    with open(features_filename, 'w') as f:
        for feature in important_features:
            f.write(f"{feature}\n")
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Scaler saved: {scaler_filename}")
    print(f"✅ Imputer saved: {imputer_filename}")
    print(f"✅ Features saved: {features_filename}")
    
    return model_filename, scaler_filename, imputer_filename, features_filename

def create_submission(model, scaler, imputer, important_features):
    """Create submission file using the trained model"""
    print(f"\n📝 CREATING SUBMISSION FILE")
    print("=" * 50)
    
    try:
        # Load test data
        print("🔄 Loading test data...")
        test_df = pd.read_parquet('test_data.parquet')
        print(f"✅ Test data loaded: {len(test_df):,} samples")
        
        # Extract features
        available_features = [f for f in important_features if f in test_df.columns]
        X_test = test_df[available_features].copy()
        
        print(f"   Using {len(available_features)} features")
        
        # Convert to numeric
        for col in X_test.columns:
            if X_test[col].dtype == 'object':
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        
        # Handle missing values
        print("   Handling missing values...")
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Scale features
        print("   Scaling features...")
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Make predictions
        print("   Making predictions...")
        predictions = model.predict_proba(X_test_scaled)[:, 1]
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id1': test_df['id1'],
            'id2': test_df['id2'],
            'id3': test_df['id3'], 
            'id5': test_df['id5'],
            'pred': predictions
        })
        
        # Save submission
        submission_filename = 'submission_balanced_nn.csv'
        submission_df.to_csv(submission_filename, index=False)
        
        print(f"✅ Submission saved: {submission_filename}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
        print(f"   Mean prediction: {predictions.mean():.4f}")
        print(f"   Columns: {submission_df.columns.tolist()}")
        
        # Show prediction distribution
        print(f"   Prediction distribution:")
        print(f"     Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
        print(f"     Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
        
        # Show quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        print(f"     Quantiles:")
        for q in quantiles:
            value = np.quantile(predictions, q)
            print(f"       {q*100:3.0f}%: {value:.4f}")
        
        return submission_filename
        
    except Exception as e:
        print(f"❌ Error creating submission: {e}")
        return None

def main():
    """Main training function"""
    print("🚀 BALANCED NEURAL NETWORK TRAINING WITH CLASS IMBALANCE HANDLING")
    print("=" * 80)
    
    start_time = time.time()
    
    # Check system resources
    available_memory = check_system_resources()
    if available_memory < 2.0:
        print("⚠️ Warning: Less than 2GB available. Consider closing other applications.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Determine sample size based on available memory
        if available_memory >= 8:
            sample_size = 200000  # 200K samples
        elif available_memory >= 6:
            sample_size = 150000  # 150K samples
        elif available_memory >= 4:
            sample_size = 100000  # 100K samples
        else:
            sample_size = 50000   # 50K samples
        
        print(f"\n🎯 TRAINING CONFIGURATION:")
        print(f"   Sample size: {sample_size:,} samples")
        print(f"   Available memory: {available_memory:.1f} GB")
        
        # Load top 100 features
        important_features = load_top_100_features()
        
        # Load and analyze data
        print(f"\n📊 Loading training data...")
        train_df = pd.read_parquet('train_data.parquet')
        
        if sample_size < len(train_df):
            train_df = train_df.sample(n=sample_size, random_state=42)
        
        print(f"✅ Training data: {len(train_df):,} samples")
        
        # Analyze class imbalance
        target_dist, imbalance_ratio = analyze_class_imbalance(train_df)
        
        # Preprocess features
        X, y, imputer = preprocess_features(train_df, important_features)
        
        # Clean up original dataframe
        del train_df
        gc.collect()
        
        # Define balancing techniques to try
        balancing_techniques = ['none', 'smote', 'random_oversample']
        
        # Add more techniques if we have enough memory
        if available_memory >= 4:
            balancing_techniques.extend(['smoteenn'])
        
        print(f"\n🎯 BALANCING TECHNIQUES TO TRY:")
        for i, technique in enumerate(balancing_techniques):
            print(f"   {i+1}. {technique.upper()}")
        
        # Train models with different balancing techniques
        results = train_neural_network_models(X, y, balancing_techniques)
        
        # Evaluate and compare models
        best_technique, best_result = evaluate_and_compare_models(results)
        
        # Save best model
        model_filename, scaler_filename, imputer_filename, features_filename = save_best_model(
            best_result, imputer, important_features
        )
        
        # Create submission file
        submission_filename = create_submission(
            best_result['model'], 
            best_result['scaler'], 
            imputer, 
            important_features
        )
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully trained balanced neural network")
        print(f"📊 Best technique: {best_technique.upper()}")
        print(f"📊 Best AUC: {best_result['auc_score']:.4f}")
        print(f"📊 Features used: {len(important_features)}")
        print(f"⏱️ Total training time: {total_time/60:.1f} minutes")
        
        if submission_filename:
            print(f"📝 Submission file: {submission_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 