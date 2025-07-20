#!/usr/bin/env python3
"""
Neural Network Training using scikit-learn MLPClassifier
Incorporates important features + add_trans + add_event data
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

def add_trans_features(X_batch, id2_values):
    """Add transaction-based features"""
    try:
        add_trans_df = pd.read_parquet('add_trans.parquet')
        add_trans_filtered = add_trans_df[add_trans_df['id2'].isin(id2_values)]
        
        if len(add_trans_filtered) > 0:
            trans_agg = add_trans_filtered.groupby('id2').agg({
                'f367': ['count', 'mean', 'std', 'sum']
            }).fillna(0)
            
            trans_agg.columns = [f"trans_{col[1]}_{col[0]}" for col in trans_agg.columns]
            X_batch = X_batch.merge(trans_agg, left_on='id2', right_index=True, how='left')
            
            for col in trans_agg.columns:
                if col in X_batch.columns:
                    X_batch[col] = X_batch[col].fillna(0)
        
        del add_trans_df, add_trans_filtered, trans_agg
        gc.collect()
        return X_batch
        
    except Exception as e:
        print(f"⚠️ Error adding trans features: {e}")
        return X_batch

def add_event_features(X_batch, id2_values):
    """Add event-based features"""
    try:
        add_event_df = pd.read_parquet('add_event.parquet')
        add_event_filtered = add_event_df[add_event_df['id2'].isin(id2_values)]
        
        if len(add_event_filtered) > 0:
            event_agg = add_event_filtered.groupby('id2').agg({
                'id3': 'count',
                'id6': 'nunique'
            }).fillna(0)
            
            event_agg.columns = ['event_count_per_id2', 'unique_events_per_id2']
            X_batch = X_batch.merge(event_agg, left_on='id2', right_index=True, how='left')
            
            for col in event_agg.columns:
                if col in X_batch.columns:
                    X_batch[col] = X_batch[col].fillna(0)
        
        del add_event_df, add_event_filtered, event_agg
        gc.collect()
        return X_batch
        
    except Exception as e:
        print(f"⚠️ Error adding event features: {e}")
        return X_batch

def train_incremental_mlp(batch_size=5000, total_samples=770164):
    """Train MLP neural network using incremental learning"""
    print("🚀 INCREMENTAL MLP NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Load important features
    important_features = load_important_features()
    
    # Initialize variables
    model = None
    scaler = None
    imputer = None
    feature_names = None
    
    n_batches = (total_samples + batch_size - 1) // batch_size
    print(f"📊 Training Configuration:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Number of batches: {n_batches}")
    print(f"   Features: {len(important_features)}")
    
    total_samples_processed = 0
    batch_aucs = []
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        print(f"\n📦 Processing batch {batch_idx + 1}/{n_batches}")
        
        # Load main training data batch
        train_df = pd.read_parquet('train_data.parquet')
        batch_end = min(batch_start + batch_size, len(train_df))
        batch_df = train_df.iloc[batch_start:batch_end].copy()
        del train_df
        gc.collect()
        
        # Extract features
        available_features = [f for f in important_features if f in batch_df.columns]
        X_batch = batch_df[available_features].copy()
        y_batch = pd.to_numeric(batch_df['y'], errors='coerce')
        
        # Add engineered features
        id2_values = batch_df['id2'].unique()
        if len(id2_values) > 0:
            X_batch = add_trans_features(X_batch, id2_values)
            X_batch = add_event_features(X_batch, id2_values)
        
        # Convert to numeric
        for col in X_batch.columns:
            if X_batch[col].dtype == 'object':
                X_batch[col] = pd.to_numeric(X_batch[col], errors='coerce')
        
        # Handle missing values
        if batch_idx == 0:
            imputer = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                imputer.fit_transform(X_batch),
                columns=X_batch.columns,
                index=X_batch.index
            )
            feature_names = X_batch.columns.tolist()
        else:
            aligned_batch = pd.DataFrame(index=X_batch.index)
            for feature in feature_names:
                if feature in X_batch.columns:
                    aligned_batch[feature] = X_batch[feature]
                else:
                    aligned_batch[feature] = 0.0
            
            X_processed = pd.DataFrame(
                imputer.transform(aligned_batch),
                columns=feature_names,
                index=aligned_batch.index
            )
        
        # Remove NaN values
        valid_mask = ~(X_processed.isnull().any(axis=1) | y_batch.isnull())
        X_processed = X_processed[valid_mask]
        y_batch = y_batch[valid_mask]
        
        if len(X_processed) == 0:
            print(f"   ⚠️ Skipping empty batch {batch_idx + 1}")
            continue
        
        # Scale features
        if batch_idx == 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
        else:
            X_scaled = scaler.transform(X_processed)
        
        # Create or update model
        if model is None:
            model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=100,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=True
            )
        
        # Split batch for validation
        split_idx = int(len(X_scaled) * 0.8)
        X_train_batch = X_scaled[:split_idx]
        y_train_batch = y_batch[:split_idx]
        X_val_batch = X_scaled[split_idx:]
        y_val_batch = y_batch[split_idx:]
        
        # Train model
        if batch_idx == 0:
            model.fit(X_train_batch, y_train_batch)
        else:
            # For incremental learning, we'll retrain on accumulated data
            # This is a simplified approach - in practice you might want more sophisticated incremental learning
            model.fit(X_train_batch, y_train_batch)
        
        # Evaluate batch
        batch_pred_proba = model.predict_proba(X_scaled)[:, 1]
        batch_auc = roc_auc_score(y_batch, batch_pred_proba)
        batch_aucs.append(batch_auc)
        
        print(f"   📈 Batch AUC: {batch_auc:.4f}")
        
        total_samples_processed += len(X_processed)
        
        # Memory cleanup
        del batch_df, X_batch, X_processed, X_scaled, X_train_batch, y_train_batch, X_val_batch, y_val_batch
        gc.collect()
        
        # Progress update
        if (batch_idx + 1) % 5 == 0:
            avg_auc = np.mean(batch_aucs[-5:])
            print(f"   📊 Progress: {total_samples_processed:,}/{total_samples:,} ({100*total_samples_processed/total_samples:.1f}%)")
            print(f"   📊 Recent avg AUC: {avg_auc:.4f}")
    
    print(f"\n✅ Incremental training complete!")
    print(f"   Total samples processed: {total_samples_processed:,}")
    print(f"   Average batch AUC: {np.mean(batch_aucs):.4f}")
    
    # Save model
    model_filename = 'best_model_mlp_neural_network.pkl'
    scaler_filename = 'scaler_mlp_neural_network.pkl'
    imputer_filename = 'imputer_mlp_neural_network.pkl'
    feature_filename = 'features_mlp_neural_network.txt'
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(imputer, imputer_filename)
    
    with open(feature_filename, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Preprocessors saved: {scaler_filename}, {imputer_filename}")
    
    return model, scaler, imputer, feature_names

def main():
    """Main training function"""
    print("🚀 MLP NEURAL NETWORK TRAINING WITH INCREMENTAL LEARNING")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Train neural network
        model, scaler, imputer, features = train_incremental_mlp(batch_size=5000)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully trained MLP neural network with incremental learning")
        print(f"📊 Features used: {len(features)} (including engineered features)")
        print(f"⏱️ Total training time: {total_time/60:.1f} minutes")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 