#!/usr/bin/env python3
"""
Enhanced Neural Network Training for AMEX Competition
Target: 0.6+ Score with probability predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
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

def load_all_features():
    """Load all available features"""
    print("🔍 Loading all available features...")
    
    # Load a small sample to get feature names
    train_df_sample = pd.read_parquet('train_data.parquet')
    train_df_sample = train_df_sample.head(1000)  # Get first 1000 rows
    feature_cols = [col for col in train_df_sample.columns if col.startswith('f')]
    
    print(f"✅ Found {len(feature_cols)} features")
    return feature_cols

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
    
    return target_dist, imbalance_ratio

def preprocess_features_enhanced(df, feature_cols):
    """Enhanced feature preprocessing"""
    print(f"\n🔧 ENHANCED FEATURE PREPROCESSING")
    print("=" * 50)
    
    # Extract features
    available_features = [f for f in feature_cols if f in df.columns]
    X = df[available_features].copy()
    y = pd.to_numeric(df['y'], errors='coerce')
    
    print(f"   Requested features: {len(feature_cols)}")
    print(f"   Available features: {len(available_features)}")
    print(f"   Missing features: {len(feature_cols) - len(available_features)}")
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

def train_enhanced_neural_network(X, y, sample_size=None):
    """Train enhanced neural network with better architecture"""
    print(f"\n🧠 ENHANCED NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Sample data if needed for memory management
    if sample_size and sample_size < len(X):
        print(f"📊 Sampling {sample_size:,} from {len(X):,} total samples")
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sampled = X.iloc[indices].copy()
        y_sampled = y.iloc[indices].copy()
    else:
        X_sampled = X
        y_sampled = y
    
    print(f"   Training on {len(X_sampled):,} samples")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_sampled, y_sampled, test_size=0.2, random_state=42, stratify=y_sampled
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    
    # Apply SMOTE balancing
    print("   Applying SMOTE balancing...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   Original training: {Counter(y_train)}")
    print(f"   Balanced training: {Counter(y_train_balanced)}")
    
    # Scale features
    print("   Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    
    # Enhanced Neural Network Architecture
    print("   Training enhanced neural network...")
    start_time = time.time()
    
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),  # Deeper network
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size=64,  # Larger batch size
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,  # More iterations
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False
    )
    
    model.fit(X_train_scaled, y_train_balanced)
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.2f}s")
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = model.predict(X_val_scaled)
    
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"   📊 Validation AUC: {auc_score:.4f}")
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_val, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   Predicted:    0      1")
    print(f"Actual 0:    {cm[0,0]:6} {cm[0,1]:6}")
    print(f"Actual 1:    {cm[1,0]:6} {cm[1,1]:6}")
    
    return model, scaler, auc_score, training_time

def create_enhanced_submission(model, scaler, imputer, feature_cols):
    """Create enhanced submission with probability predictions"""
    print(f"\n📝 CREATING ENHANCED SUBMISSION")
    print("=" * 50)
    
    try:
        # Load test data
        print("🔄 Loading test data...")
        test_df = pd.read_parquet('test_data.parquet')
        print(f"✅ Test data loaded: {len(test_df):,} samples")
        
        # Extract features
        available_features = [f for f in feature_cols if f in test_df.columns]
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
        
        # Make probability predictions
        print("   Making probability predictions...")
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
        submission_filename = 'submission_enhanced_nn.csv'
        submission_df.to_csv(submission_filename, index=False)
        
        print(f"✅ Enhanced submission saved: {submission_filename}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
        print(f"   Mean prediction: {predictions.mean():.4f}")
        
        # Show prediction distribution
        print(f"\n📊 Prediction Distribution:")
        print(f"   Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
        print(f"   Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
        
        # Show quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        print(f"   Quantiles:")
        for q in quantiles:
            value = np.quantile(predictions, q)
            print(f"     {q*100:3.0f}%: {value:.4f}")
        
        return submission_filename
        
    except Exception as e:
        print(f"❌ Error creating submission: {e}")
        return None

def main():
    """Main enhanced training function"""
    print("🚀 ENHANCED NEURAL NETWORK TRAINING FOR AMEX COMPETITION")
    print("=" * 80)
    
    start_time = time.time()
    
    # Check system resources
    available_memory = check_system_resources()
    
    try:
        # Determine sample size based on available memory
        if available_memory >= 8:
            sample_size = 300000  # 300K samples
        elif available_memory >= 6:
            sample_size = 200000  # 200K samples
        elif available_memory >= 4:
            sample_size = 150000  # 150K samples
        else:
            sample_size = 100000  # 100K samples
        
        print(f"\n🎯 TRAINING CONFIGURATION:")
        print(f"   Sample size: {sample_size:,} samples")
        print(f"   Available memory: {available_memory:.1f} GB")
        
        # Load all features
        feature_cols = load_all_features()
        
        # Load training data
        print(f"\n📊 Loading training data...")
        train_df = pd.read_parquet('train_data.parquet')
        
        if sample_size < len(train_df):
            train_df = train_df.sample(n=sample_size, random_state=42)
        
        print(f"✅ Training data: {len(train_df):,} samples")
        
        # Analyze class imbalance
        target_dist, imbalance_ratio = analyze_class_imbalance(train_df)
        
        # Preprocess features
        X, y, imputer = preprocess_features_enhanced(train_df, feature_cols)
        
        # Clean up original dataframe
        del train_df
        gc.collect()
        
        # Train enhanced model
        model, scaler, auc_score, training_time = train_enhanced_neural_network(X, y, sample_size)
        
        # Save enhanced model
        print(f"\n💾 SAVING ENHANCED MODEL")
        print("=" * 40)
        
        model_filename = 'best_model_enhanced_nn.pkl'
        scaler_filename = 'scaler_enhanced_nn.pkl'
        imputer_filename = 'imputer_enhanced_nn.pkl'
        features_filename = 'features_enhanced_nn.txt'
        
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(imputer, imputer_filename)
        
        with open(features_filename, 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        print(f"✅ Model saved: {model_filename}")
        print(f"✅ Scaler saved: {scaler_filename}")
        print(f"✅ Imputer saved: {imputer_filename}")
        print(f"✅ Features saved: {features_filename}")
        
        # Create enhanced submission
        submission_filename = create_enhanced_submission(model, scaler, imputer, feature_cols)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 ENHANCED TRAINING COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully trained enhanced neural network")
        print(f"📊 Validation AUC: {auc_score:.4f}")
        print(f"📊 Features used: {len(feature_cols)}")
        print(f"📊 Training time: {training_time:.2f}s")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        
        if submission_filename:
            print(f"📝 Enhanced submission file: {submission_filename}")
        
        # Expected score improvement
        print(f"\n🎯 EXPECTED SCORE IMPROVEMENT:")
        print(f"   Previous score: ~0.2")
        print(f"   Expected score: 0.5-0.7")
        print(f"   Improvement: +0.3-0.5 points")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 