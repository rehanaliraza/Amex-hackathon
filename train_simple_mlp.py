#!/usr/bin/env python3
"""
Simple MLP Neural Network Training
Uses important features with basic feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
import time
import psutil
import joblib
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

def load_important_features():
    """Load the most important features"""
    try:
        with open('top_100_features.txt', 'r') as f:
            features = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(features)} important features")
    except FileNotFoundError:
        features = ['f366', 'f223', 'f132', 'f363', 'f210', 'f365', 'f364', 'f362']
        print(f"⚠️ Using fallback {len(features)} features")
    return features

def train_simple_mlp():
    """Train simple MLP neural network"""
    print("🚀 SIMPLE MLP NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load important features
        important_features = load_important_features()
        
        # Load training data (use 10% for faster training)
        print("\n📊 Loading training data...")
        train_df = pd.read_parquet('train_data.parquet')
        
        # Use 10% of data for faster training
        sample_size = min(50000, len(train_df))
        train_df = train_df.sample(n=sample_size, random_state=42)
        print(f"✅ Loaded {len(train_df):,} samples")
        
        # Extract features
        available_features = [f for f in important_features if f in train_df.columns]
        X = train_df[available_features].copy()
        y = pd.to_numeric(train_df['y'], errors='coerce')
        
        print(f"✅ Using {len(available_features)} features")
        
        # Convert to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Remove NaN values
        valid_mask = ~(X_imputed.isnull().any(axis=1) | y.isnull())
        X_clean = X_imputed[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"✅ Clean data: {len(X_clean):,} samples, {len(X_clean.columns)} features")
        
        # Split data
        split_idx = int(len(X_clean) * 0.8)
        X_train = X_clean.iloc[:split_idx]
        y_train = y_clean.iloc[:split_idx]
        X_val = X_clean.iloc[split_idx:]
        y_val = y_clean.iloc[split_idx:]
        
        print(f"✅ Training samples: {len(X_train):,}")
        print(f"✅ Validation samples: {len(X_val):,}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create MLP neural network
        print("\n🤖 Creating MLP neural network...")
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True
        )
        
        print("✅ MLP neural network created")
        
        # Train model
        print("\n🔄 Training MLP neural network...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"✅ Validation AUC: {val_auc:.4f}")
        
        # Save model
        print("\n💾 Saving model...")
        model_filename = 'best_model_simple_mlp.pkl'
        scaler_filename = 'scaler_simple_mlp.pkl'
        imputer_filename = 'imputer_simple_mlp.pkl'
        feature_filename = 'features_simple_mlp.txt'
        
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(imputer, imputer_filename)
        
        with open(feature_filename, 'w') as f:
            for feature in available_features:
                f.write(f"{feature}\n")
        
        print(f"✅ Model saved: {model_filename}")
        print(f"✅ Preprocessors saved: {scaler_filename}, {imputer_filename}")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully trained MLP neural network")
        print(f"📊 Features used: {len(available_features)}")
        print(f"📈 Validation AUC: {val_auc:.4f}")
        print(f"⏱️ Total training time: {total_time/60:.1f} minutes")
        print(f"💾 Model saved: {model_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    print("🚀 SIMPLE MLP NEURAL NETWORK TRAINING")
    print("=" * 80)
    
    train_simple_mlp()

if __name__ == "__main__":
    main() 