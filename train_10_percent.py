#!/usr/bin/env python3
"""
Train ML Model on 10% of AMEX Dataset
Improved training with larger dataset for better performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
import time
import psutil
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

def check_system_resources():
    """Check if system has enough resources for 10% training"""
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
    
    # Estimate memory needed for 10% training
    total_samples = 770164
    target_samples = int(total_samples * 0.10)
    memory_per_row = 17784  # From previous analysis
    estimated_memory = (target_samples * memory_per_row) / (1024**3)
    
    print(f"\n📊 Training Plan:")
    print(f"   Target samples: {target_samples:,} (10%)")
    print(f"   Estimated memory: {estimated_memory:.1f} GB")
    
    if available_gb >= estimated_memory + 0.5:  # +0.5GB buffer
        print(f"   ✅ Sufficient memory available")
        return True
    else:
        print(f"   ⚠️ May need to close other applications")
        return available_gb >= estimated_memory * 0.8  # Still try if close

def load_and_preprocess_data(sample_size=77000):
    """Load and preprocess 10% of the AMEX data"""
    print(f"\n📊 LOADING 10% DATA (sample size: {sample_size:,})")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load training data
    print("🔄 Loading training data...")
    train_df = pd.read_parquet('train_data.parquet')
    
    # Sample 10% of the data
    print(f"📊 Sampling {sample_size:,} from {len(train_df):,} total rows")
    train_df = train_df.sample(n=sample_size, random_state=42)
    
    print(f"✅ Training data: {train_df.shape[0]:,} × {train_df.shape[1]} columns")
    print(f"⏱️ Load time: {time.time() - start_time:.2f} seconds")
    
    # Analyze target
    print(f"\n🎯 TARGET ANALYSIS:")
    target_dist = train_df['y'].value_counts()
    for class_val, count in target_dist.items():
        pct = (count / len(train_df)) * 100
        print(f"   Class {class_val}: {count:,} ({pct:.1f}%)")
    
    # Prepare features
    print(f"\n🔧 PREPROCESSING:")
    feature_cols = [col for col in train_df.columns if col.startswith('f')]
    X = train_df[feature_cols].copy()
    y = pd.to_numeric(train_df['y'], errors='coerce')
    
    print(f"   Initial features: {len(feature_cols)}")
    print(f"   Initial missing values: {X.isnull().sum().sum():,}")
    
    # Convert object columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Remove columns that are entirely NaN after conversion
    valid_cols = []
    for col in X.columns:
        if X[col].notna().sum() > 0:
            valid_cols.append(col)
    
    X_clean = X[valid_cols].copy()
    print(f"   Features after cleaning: {len(valid_cols)} (removed {len(feature_cols) - len(valid_cols)} empty columns)")
    print(f"   Missing values after cleaning: {X_clean.isnull().sum().sum():,}")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )
    
    print(f"   After imputation: {X_imputed.isnull().sum().sum()} missing values")
    print(f"✅ Preprocessing complete")
    
    return X_imputed, y, imputer

def train_and_evaluate_models(X, y):
    """Train multiple models and evaluate performance"""
    print(f"\n🤖 MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split: {X_train.shape[0]:,} train, {X_val.shape[0]:,} validation")
    
    # Define models with better parameters for larger dataset
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', 
            n_jobs=-1, max_depth=10
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42, learning_rate=0.1, max_depth=6
        )
    }
    
    results = {}
    best_auc = 0
    best_model_name = ""
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        training_time = time.time() - start_time
        
        results[name] = {
            'model': model,
            'auc_score': auc_score,
            'training_time': training_time,
            'predictions': y_pred_proba
        }
        
        print(f"   ✅ AUC: {auc_score:.4f}, Time: {training_time:.2f}s")
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_model_name = name
    
    print(f"\n🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")
    
    return results, best_model_name, X_train, X_val, y_train, y_val

def cross_validate_best_model(best_model, X, y, cv_folds=5):
    """Perform cross-validation on the best model"""
    print(f"\n🔄 CROSS-VALIDATION ({cv_folds}-fold)")
    print("=" * 50)
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"CV AUC Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return cv_scores

def analyze_feature_importance(model, feature_names, top_n=25):
    """Analyze feature importance if available"""
    if hasattr(model, 'feature_importances_'):
        print(f"\n🔍 TOP {top_n} FEATURE IMPORTANCES")
        print("=" * 60)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(top_n)
        
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{idx:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Calculate cumulative importance
        cumsum_importance = np.cumsum(importance_df['importance'])
        features_80_pct = np.argmax(cumsum_importance >= 0.8) + 1
        print(f"\n📊 {features_80_pct} features contribute to 80% of importance")
        
        return importance_df
    else:
        print(f"\nℹ️ Feature importance not available for this model type")
        return None

def save_model_and_results(best_model, imputer, model_name):
    """Save the trained model and preprocessing pipeline"""
    print(f"\n💾 SAVING MODEL AND RESULTS")
    print("=" * 50)
    
    # Save model with 10% suffix
    model_filename = f'best_model_10pct_{model_name.lower().replace(" ", "_")}.pkl'
    preprocessor_filename = 'data_preprocessor_10pct.pkl'
    
    joblib.dump(best_model, model_filename)
    joblib.dump(imputer, preprocessor_filename)
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Preprocessor saved: {preprocessor_filename}")
    
    return model_filename, preprocessor_filename

def main():
    """Main training function"""
    print("🚀 AMEX 10% DATASET TRAINING")
    print("=" * 70)
    
    # Check system resources
    if not check_system_resources():
        print("⚠️ Warning: Limited system resources. Training may be slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Load and preprocess data (10% = ~77,000 samples)
        X, y, imputer = load_and_preprocess_data(sample_size=77000)
        
        # Train and evaluate models
        results, best_model_name, X_train, X_val, y_train, y_val = train_and_evaluate_models(X, y)
        
        # Get best model
        best_model = results[best_model_name]['model']
        
        # Cross-validation
        cv_scores = cross_validate_best_model(best_model, X, y)
        
        # Feature importance analysis
        importance_df = analyze_feature_importance(best_model, X.columns)
        
        # Save model
        model_file, preprocessor_file = save_model_and_results(best_model, imputer, best_model_name)
        
        # Final summary
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 70)
        print(f"✅ Successfully completed 10% dataset training")
        print(f"📊 Dataset: {len(X):,} samples, {len(X.columns)} features")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📈 Validation AUC: {results[best_model_name]['auc_score']:.4f}")
        print(f"📈 Cross-validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"⏱️ Total training time: {sum(r['training_time'] for r in results.values()):.2f}s")
        
        # System performance
        memory_info = psutil.virtual_memory()
        print(f"\n💻 System Performance:")
        print(f"   Memory usage: {memory_info.percent:.1f}%")
        print(f"   Available memory: {memory_info.available / (1024**3):.1f} GB")
        
        print(f"\n🎉 10% TRAINING COMPLETE!")
        print(f"   Model file: {model_file}")
        print(f"   Preprocessor: {preprocessor_file}")
        print(f"   Ready to create improved submission!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 Next step: Run create_submission.py with the new model!")
    else:
        print(f"\n⚠️ Training incomplete - check system resources") 