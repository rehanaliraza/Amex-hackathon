#!/usr/bin/env python3
"""
Simple Neural Network Training with Incremental Learning
Incorporates important features + add_trans + add_event data
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import time
import psutil
import joblib
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

def main():
    """Main training function"""
    print("🚀 SIMPLE NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load important features
        try:
            with open('top_100_features.txt', 'r') as f:
                important_features = [line.strip() for line in f.readlines()][:20]  # Use top 20
            print(f"✅ Loaded {len(important_features)} important features")
        except FileNotFoundError:
            important_features = ['f366', 'f223', 'f132', 'f363', 'f210', 'f365', 'f364', 'f362']
            print(f"⚠️ Using fallback {len(important_features)} features")
        
        # Load sample data for training
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create neural network
        print("\n🤖 Creating neural network...")
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        print("✅ Neural network created")
        model.summary()
        
        # Train model
        print("\n🔄 Training neural network...")
        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        y_pred_proba = model.predict(X_val_scaled)
        val_auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"✅ Validation AUC: {val_auc:.4f}")
        
        # Save model
        print("\n💾 Saving model...")
        model_filename = 'best_model_simple_neural_network.h5'
        scaler_filename = 'scaler_simple_neural_network.pkl'
        imputer_filename = 'imputer_simple_neural_network.pkl'
        feature_filename = 'features_simple_neural_network.txt'
        
        model.save(model_filename)
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
        print(f"✅ Successfully trained neural network")
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

if __name__ == "__main__":
    main() 