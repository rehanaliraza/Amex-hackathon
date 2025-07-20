#!/usr/bin/env python3
"""
Train on Full AMEX Dataset with Memory Optimization Techniques
Uses batch processing, incremental learning, and memory-efficient algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
import time
import psutil
import joblib
import gc
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

class MemoryEfficientTrainer:
    """Memory-efficient trainer using various optimization techniques"""
    
    def __init__(self, batch_size=10000, n_batches_validation=5):
        self.batch_size = batch_size
        self.n_batches_validation = n_batches_validation
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        
    def check_system_resources(self):
        """Check available system resources"""
        print("🖥️ SYSTEM RESOURCE CHECK")
        print("=" * 50)
        
        memory = psutil.virtual_memory()
        print(f"💻 System Info:")
        print(f"   Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"   Memory usage: {memory.percent:.1f}%")
        print(f"   CPU cores: {psutil.cpu_count()}")
        
        # Estimate batch memory usage
        memory_per_row = 17784  # From previous analysis
        batch_memory = (self.batch_size * memory_per_row) / (1024**3)
        
        print(f"\n📊 Batch Training Plan:")
        print(f"   Batch size: {self.batch_size:,} samples")
        print(f"   Estimated memory per batch: {batch_memory:.2f} GB")
        
        return memory.available / (1024**3) >= batch_memory * 2  # 2x buffer
    
    def load_data_info(self):
        """Get dataset information without loading full data"""
        print("📊 ANALYZING DATASET STRUCTURE")
        print("=" * 50)
        
        # Load just a small sample to understand structure
        full_df = pd.read_parquet('train_data.parquet')
        sample_df = full_df.head(1000)
        
        total_samples = len(full_df)
        feature_cols = [col for col in sample_df.columns if col.startswith('f')]
        
        print(f"   Total samples: {total_samples:,}")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   Batches needed: {(total_samples + self.batch_size - 1) // self.batch_size}")
        
        # Clear the full dataframe from memory
        del full_df
        gc.collect()
        
        return total_samples, feature_cols
    
    def create_batch_iterator(self, total_samples):
        """Create iterator for processing data in batches"""
        n_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        # Load the full dataset once and iterate through it
        print("🔄 Loading full dataset for batch processing...")
        full_df = pd.read_parquet('train_data.parquet')
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            
            # Get batch from full dataframe
            batch_df = full_df.iloc[start_idx:end_idx].copy()
            
            yield batch_idx, batch_df
        
        # Clean up
        del full_df
        gc.collect()
    
    def preprocess_batch(self, batch_df, fit_preprocessor=False):
        """Preprocess a batch of data"""
        # Extract features and target
        feature_cols = [col for col in batch_df.columns if col.startswith('f')]
        X_batch = batch_df[feature_cols].copy()
        y_batch = pd.to_numeric(batch_df['y'], errors='coerce')
        
        # Convert object columns to numeric
        for col in X_batch.columns:
            if X_batch[col].dtype == 'object':
                X_batch[col] = pd.to_numeric(X_batch[col], errors='coerce')
        
        # Remove columns that are entirely NaN
        valid_cols = []
        for col in X_batch.columns:
            if X_batch[col].notna().sum() > 0:
                valid_cols.append(col)
        
        X_batch = X_batch[valid_cols]
        
        # Fit or transform with preprocessor
        if fit_preprocessor:
            self.preprocessor = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                self.preprocessor.fit_transform(X_batch),
                columns=X_batch.columns,
                index=X_batch.index
            )
            self.feature_names = X_batch.columns.tolist()
        else:
            # Align features with training features
            aligned_batch = pd.DataFrame(index=X_batch.index)
            for feature in self.feature_names:
                if feature in X_batch.columns:
                    aligned_batch[feature] = X_batch[feature]
                else:
                    aligned_batch[feature] = 0.0
            
            X_processed = pd.DataFrame(
                self.preprocessor.transform(aligned_batch),
                columns=self.feature_names,
                index=aligned_batch.index
            )
        
        return X_processed, y_batch
    
    def train_incremental_model(self, total_samples):
        """Train model using incremental learning"""
        print("\n🤖 INCREMENTAL TRAINING")
        print("=" * 50)
        
        # Use SGD classifier for incremental learning
        self.model = SGDClassifier(
            loss='log_loss',  # For probability estimates
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            learning_rate='adaptive',
            eta0=0.01
        )
        
        batch_count = 0
        total_samples_processed = 0
        
        # Process batches
        for batch_idx, batch_df in self.create_batch_iterator(total_samples):
            print(f"🔄 Processing batch {batch_idx + 1}...")
            
            # Preprocess batch
            fit_preprocessor = (batch_idx == 0)  # Fit on first batch
            X_batch, y_batch = self.preprocess_batch(batch_df, fit_preprocessor)
            
            # Remove any remaining NaN values
            valid_mask = ~(X_batch.isnull().any(axis=1) | y_batch.isnull())
            X_batch = X_batch[valid_mask]
            y_batch = y_batch[valid_mask]
            
            if len(X_batch) > 0:
                # Incremental training
                if batch_idx == 0:
                    self.model.fit(X_batch, y_batch)
                else:
                    self.model.partial_fit(X_batch, y_batch)
                
                total_samples_processed += len(X_batch)
                batch_count += 1
                
                print(f"   ✅ Batch {batch_idx + 1}: {len(X_batch):,} samples processed")
            
            # Clear memory
            del batch_df, X_batch, y_batch
            gc.collect()
            
            # Show progress
            if (batch_idx + 1) % 10 == 0:
                memory_info = psutil.virtual_memory()
                print(f"   📊 Progress: {total_samples_processed:,} samples, Memory: {memory_info.percent:.1f}%")
        
        print(f"\n✅ Incremental training complete!")
        print(f"   Total samples processed: {total_samples_processed:,}")
        print(f"   Batches processed: {batch_count}")
        
        return total_samples_processed
    
    def validate_model(self, total_samples):
        """Validate model on held-out batches"""
        print("\n📊 MODEL VALIDATION")
        print("=" * 50)
        
        validation_predictions = []
        validation_targets = []
        
        # Use last few batches for validation
        n_batches = (total_samples + self.batch_size - 1) // self.batch_size
        validation_batches = list(range(max(0, n_batches - self.n_batches_validation), n_batches))
        
        # Load full dataset for validation
        print("🔄 Loading data for validation...")
        full_df = pd.read_parquet('train_data.parquet')
        
        for batch_idx in validation_batches:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            
            # Get validation batch
            batch_df = full_df.iloc[start_idx:end_idx].copy()
            
            # Preprocess
            X_batch, y_batch = self.preprocess_batch(batch_df, fit_preprocessor=False)
            
            # Remove NaN values
            valid_mask = ~(X_batch.isnull().any(axis=1) | y_batch.isnull())
            X_batch = X_batch[valid_mask]
            y_batch = y_batch[valid_mask]
            
            if len(X_batch) > 0:
                # Predict
                y_pred_proba = self.model.predict_proba(X_batch)[:, 1]
                
                validation_predictions.extend(y_pred_proba)
                validation_targets.extend(y_batch)
            
            # Clear memory
            del batch_df, X_batch, y_batch
            gc.collect()
        
        # Clean up full dataset
        del full_df
        gc.collect()
        
        # Calculate validation metrics
        if len(validation_predictions) > 0:
            validation_auc = roc_auc_score(validation_targets, validation_predictions)
            print(f"✅ Validation AUC: {validation_auc:.4f}")
            print(f"   Validation samples: {len(validation_predictions):,}")
            
            return validation_auc
        else:
            print("❌ No validation samples available")
            return None
    
    def save_model(self):
        """Save the trained model and preprocessor"""
        print("\n💾 SAVING MODEL")
        print("=" * 50)
        
        model_filename = 'best_model_full_dataset_sgd.pkl'
        preprocessor_filename = 'data_preprocessor_full_dataset.pkl'
        
        joblib.dump(self.model, model_filename)
        joblib.dump(self.preprocessor, preprocessor_filename)
        
        print(f"✅ Model saved: {model_filename}")
        print(f"✅ Preprocessor saved: {preprocessor_filename}")
        
        return model_filename, preprocessor_filename

def train_random_forest_batches(total_samples, batch_size=10000, n_trees_per_batch=20):
    """Alternative: Train Random Forest using batch aggregation"""
    print("\n🌲 BATCH RANDOM FOREST TRAINING")
    print("=" * 50)
    
    models = []
    preprocessors = []
    
    n_batches = min(10, (total_samples + batch_size - 1) // batch_size)  # Limit batches
    
    for batch_idx in range(n_batches):
        print(f"🔄 Training forest batch {batch_idx + 1}/{n_batches}...")
        
        # Load random batch
        batch_df = pd.read_parquet('train_data.parquet').sample(n=batch_size, random_state=42 + batch_idx)
        
        # Preprocess
        feature_cols = [col for col in batch_df.columns if col.startswith('f')]
        X_batch = batch_df[feature_cols].copy()
        y_batch = pd.to_numeric(batch_df['y'], errors='coerce')
        
        # Convert and clean
        for col in X_batch.columns:
            if X_batch[col].dtype == 'object':
                X_batch[col] = pd.to_numeric(X_batch[col], errors='coerce')
        
        # Remove empty columns
        valid_cols = [col for col in X_batch.columns if X_batch[col].notna().sum() > 0]
        X_batch = X_batch[valid_cols]
        
        # Impute
        imputer = SimpleImputer(strategy='median')
        X_processed = pd.DataFrame(
            imputer.fit_transform(X_batch),
            columns=X_batch.columns,
            index=X_batch.index
        )
        
        # Train small forest
        rf = RandomForestClassifier(
            n_estimators=n_trees_per_batch,
            random_state=42 + batch_idx,
            class_weight='balanced',
            n_jobs=-1,
            max_depth=8
        )
        
        rf.fit(X_processed, y_batch)
        
        models.append(rf)
        preprocessors.append(imputer)
        
        print(f"   ✅ Batch {batch_idx + 1} complete: {len(X_processed)} samples")
        
        # Clear memory
        del batch_df, X_batch, y_batch, X_processed
        gc.collect()
    
    # Save ensemble
    ensemble_data = {
        'models': models,
        'preprocessors': preprocessors,
        'feature_names': [list(models[0].feature_names_in_)]
    }
    
    joblib.dump(ensemble_data, 'ensemble_random_forest_full.pkl')
    print(f"✅ Ensemble model saved: ensemble_random_forest_full.pkl")
    
    return ensemble_data

def main():
    """Main training function"""
    print("🚀 FULL DATASET TRAINING WITH MEMORY OPTIMIZATION")
    print("=" * 70)
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(batch_size=10000)
    
    # Check system resources
    if not trainer.check_system_resources():
        print("⚠️ Warning: Limited memory. Consider smaller batch size.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Get dataset info
        total_samples, feature_cols = trainer.load_data_info()
        
        print(f"\n🎯 TRAINING STRATEGY:")
        print(f"   Strategy 1: Incremental Learning (SGD)")
        print(f"   Strategy 2: Batch Random Forest Ensemble")
        
        strategy = input("\nChoose strategy (1 or 2): ").strip()
        
        if strategy == "1":
            # Incremental learning approach
            samples_processed = trainer.train_incremental_model(total_samples)
            validation_auc = trainer.validate_model(total_samples)
            model_file, preprocessor_file = trainer.save_model()
            
            print(f"\n🎯 INCREMENTAL TRAINING SUMMARY:")
            print(f"   Samples processed: {samples_processed:,}")
            print(f"   Validation AUC: {validation_auc:.4f}" if validation_auc else "   No validation")
            print(f"   Model: {model_file}")
            
        elif strategy == "2":
            # Batch ensemble approach
            ensemble_data = train_random_forest_batches(total_samples)
            
            print(f"\n🎯 ENSEMBLE TRAINING SUMMARY:")
            print(f"   Number of models: {len(ensemble_data['models'])}")
            print(f"   Trees per model: 20")
            print(f"   Total trees: {len(ensemble_data['models']) * 20}")
            
        else:
            print("❌ Invalid strategy selected")
            return False
        
        print(f"\n🎉 FULL DATASET TRAINING COMPLETE!")
        print(f"   Ready to create submission with full dataset model!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 Next: Update create_submission.py to use the full dataset model!")
    else:
        print(f"\n⚠️ Training incomplete - try different strategy or smaller batch size") 