#!/usr/bin/env python3
"""
Create Submission File for AMEX Dataset
Uses trained model to generate predictions on test data
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

def load_trained_model():
    """Load the trained model and preprocessor"""
    print("🔄 Loading trained model and preprocessor...")
    
    # Check which models are available (prioritize XGBoost)
    models_available = []
    
    # Check for XGBoost model first (highest priority)
    if Path('best_model_full_xgboost_optimized.json').exists():
        models_available.append(('Full XGBoost (AUC: 0.89)', 'best_model_full_xgboost_optimized.json', 'data_preprocessor_full_xgboost.pkl', 'xgboost'))
    
    # Check for Auto Random Forest model
    if Path('best_model_auto_rf_balanced.pkl').exists():
        models_available.append(('Auto Random Forest (AUC: 0.92)', 'best_model_auto_rf_balanced.pkl', 'data_preprocessor_auto_rf_balanced.pkl', 'sklearn'))
    
    # Check for other Random Forest models
    if Path('best_model_simple_rf_balanced.pkl').exists():
        models_available.append(('Random Forest (Balanced)', 'best_model_simple_rf_balanced.pkl', 'data_preprocessor_simple_rf.pkl', 'sklearn'))
    
    # Check for other models
    if Path('best_model_10pct_gradient_boosting.pkl').exists():
        models_available.append(('10% Gradient Boosting', 'best_model_10pct_gradient_boosting.pkl', 'data_preprocessor_10pct.pkl', 'sklearn'))
    if Path('best_model_full_simple_sgd.pkl').exists():
        models_available.append(('Full Dataset SGD', 'best_model_full_simple_sgd.pkl', 'data_preprocessor_full_simple.pkl', 'sklearn'))
    
    if not models_available:
        print("❌ No trained models found!")
        print("💡 Run train_full_xgboost_optimized.py, train_rf_auto.py, or other training scripts first!")
        return None, None, None
    
    # Show available models
    print("📊 Available models:")
    for i, (name, _, _, _) in enumerate(models_available):
        print(f"   {i+1}. {name}")
    
    # Auto-select the first (highest priority) model
    selected_idx = 0
    model_name, model_file, preprocessor_file, model_type = models_available[selected_idx]
    print(f"✅ Using: {model_name}")
    
    try:
        # Load model based on type
        if model_type == 'xgboost':
            model = xgb.Booster()
            model.load_model(model_file)
        else:
            model = joblib.load(model_file)
        
        preprocessor = joblib.load(preprocessor_file)
        
        print(f"✅ Model loaded successfully: {model_name}")
        return model, preprocessor, model_type
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def load_feature_list_for_xgboost():
    """Load feature list for XGBoost model"""
    feature_files = ['features_full_xgboost.txt', 'tier1_tier2_features.txt']
    
    for filename in feature_files:
        if Path(filename).exists():
            try:
                with open(filename, 'r') as f:
                    features = [line.strip() for line in f.readlines() if line.strip()]
                print(f"✅ Loaded {len(features)} features from {filename}")
                return features
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
    
    print("⚠️ No feature list found, using default feature extraction")
    return None

def load_test_data():
    """Load test data efficiently"""
    print("🔄 Loading test data...")
    
    try:
        # Load test data
        test_df = pd.read_parquet('test_data.parquet')
        print(f"✅ Test data loaded: {len(test_df):,} rows × {len(test_df.columns)} columns")
        
        return test_df
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

def preprocess_test_data(test_df, preprocessor, model_type, feature_list=None):
    """Preprocess test data using the same pipeline as training"""
    print("🔧 Preprocessing test data...")
    
    if model_type == 'xgboost' and feature_list:
        # Use specific feature list for XGBoost
        print(f"   Using XGBoost feature list: {len(feature_list)} features")
        feature_cols = [f for f in feature_list if f in test_df.columns]
    else:
        # Extract all feature columns (same as training)
        feature_cols = [col for col in test_df.columns if col.startswith('f')]
    
    X_test = test_df[feature_cols].copy()
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Missing values: {X_test.isnull().sum().sum():,}")
    
    # Convert object columns to numeric (same as training)
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Remove columns that are entirely NaN after conversion
    valid_cols = []
    for col in X_test.columns:
        if X_test[col].notna().sum() > 0:
            valid_cols.append(col)
    
    X_test_clean = X_test[valid_cols].copy()
    print(f"   Features after cleaning: {len(valid_cols)}")
    
    # Get the exact feature names that the preprocessor was trained on
    if hasattr(preprocessor, 'feature_names_in_'):
        expected_features = list(preprocessor.feature_names_in_)
        print(f"   Expected features from training: {len(expected_features)}")
        
        # Create a DataFrame with the exact features expected by the model
        X_test_aligned = pd.DataFrame(index=X_test_clean.index)
        
        missing_features = []
        for feature in expected_features:
            if feature in X_test_clean.columns:
                X_test_aligned[feature] = X_test_clean[feature]
            else:
                X_test_aligned[feature] = 0.0  # Fill missing features with 0
                missing_features.append(feature)
        
        if missing_features:
            print(f"   Missing features filled with 0: {len(missing_features)}")
            if len(missing_features) <= 10:
                print(f"   Missing: {missing_features}")
            else:
                print(f"   Missing: {missing_features[:5]} ... {missing_features[-5:]}")
        
        print(f"   Aligned features: {len(X_test_aligned.columns)}")
        
    else:
        print("   Warning: Could not get training feature names, using available features")
        X_test_aligned = X_test_clean
    
    # Apply the same preprocessing pipeline
    try:
        X_test_processed = pd.DataFrame(
            preprocessor.transform(X_test_aligned),
            columns=X_test_aligned.columns,
            index=X_test_aligned.index
        )
        
        print(f"   After preprocessing: {X_test_processed.isnull().sum().sum()} missing values")
        print("✅ Test data preprocessing complete")
        
        return X_test_processed
        
    except Exception as e:
        print(f"❌ Error preprocessing test data: {e}")
        return None

def generate_predictions(model, X_test_processed, model_type):
    """Generate predictions using the trained model"""
    print("🔮 Generating predictions...")
    
    try:
        if model_type == 'xgboost':
            # XGBoost prediction
            dtest = xgb.DMatrix(X_test_processed)
            predictions_proba = model.predict(dtest)
            
            # Convert probabilities to binary predictions using 0.5 threshold
            predictions_binary = (predictions_proba >= 0.5).astype(int)
        else:
            # Scikit-learn prediction
            predictions_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # Convert probabilities to binary predictions using 0.5 threshold
            predictions_binary = (predictions_proba >= 0.5).astype(int)
        
        print(f"✅ Predictions generated for {len(predictions_binary):,} samples")
        print(f"   Probability range: {predictions_proba.min():.4f} to {predictions_proba.max():.4f}")
        print(f"   Mean probability: {predictions_proba.mean():.4f}")
        print(f"   Binary predictions: {predictions_binary.sum():,} positive (1) out of {len(predictions_binary):,} total")
        print(f"   Positive rate: {predictions_binary.mean():.4f} ({predictions_binary.mean()*100:.2f}%)")
        
        return predictions_binary
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return None

def create_submission_file(test_df, predictions, output_filename='submission.csv'):
    """Create submission file in the required format"""
    print("📝 Creating submission file...")
    
    try:
        # Load submission template to understand the format
        template_df = pd.read_csv('685404e30cfdb_submission_template.csv')
        print(f"   Template format: {list(template_df.columns)}")
        
        # Extract ID columns from test data
        id_cols = ['id1', 'id2', 'id3', 'id5']
        
        # Create submission dataframe
        submission_df = pd.DataFrame()
        
        # Copy ID columns from test data
        for col in id_cols:
            if col in test_df.columns:
                submission_df[col] = test_df[col]
            else:
                print(f"⚠️ Warning: Column {col} not found in test data")
        
        # Add predictions
        submission_df['pred'] = predictions
        
        # Save submission file
        submission_df.to_csv(output_filename, index=False)
        
        print(f"✅ Submission file created: {output_filename}")
        print(f"   Shape: {submission_df.shape}")
        print(f"   Columns: {list(submission_df.columns)}")
        
        # Show sample of submission
        print(f"\n📊 Sample of submission file:")
        print(submission_df.head())
        
        # Show prediction statistics
        print(f"\n📈 Prediction Statistics:")
        print(f"   Min: {predictions.min():.6f}")
        print(f"   Max: {predictions.max():.6f}")
        print(f"   Mean: {predictions.mean():.6f}")
        print(f"   Std: {predictions.std():.6f}")
        
        return submission_df
        
    except Exception as e:
        print(f"❌ Error creating submission file: {e}")
        return None

def validate_submission(submission_df):
    """Validate the submission file format"""
    print("🔍 Validating submission file...")
    
    required_cols = ['id1', 'id2', 'id3', 'id5', 'pred']
    
    # Check columns
    missing_cols = [col for col in required_cols if col not in submission_df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    
    # Check for missing values
    null_counts = submission_df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"⚠️ Warning: Found missing values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"   {col}: {count} missing values")
    
    # Check that predictions are binary (0 or 1)
    unique_preds = set(submission_df['pred'].unique())
    if not unique_preds.issubset({0, 1}):
        print(f"❌ Error: Predictions must be binary (0 or 1). Found: {sorted(unique_preds)}")
        return False
    
    # Show prediction distribution
    pred_counts = submission_df['pred'].value_counts().sort_index()
    print(f"   Prediction distribution:")
    for pred_val, count in pred_counts.items():
        pct = (count / len(submission_df)) * 100
        print(f"     {pred_val}: {count:,} ({pct:.2f}%)")
    
    print("✅ Validation complete")
    return True

def main():
    """Main submission creation function"""
    print("🚀 AMEX SUBMISSION FILE CREATION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load trained model
    model, preprocessor, model_type = load_trained_model()
    if model is None or preprocessor is None:
        print("❌ Failed to load model. Please run training first.")
        return
    
    # Load feature list if XGBoost
    feature_list = None
    if model_type == 'xgboost':
        feature_list = load_feature_list_for_xgboost()
    
    # Load test data
    test_df = load_test_data()
    if test_df is None:
        print("❌ Failed to load test data.")
        return
    
    # Preprocess test data
    X_test_processed = preprocess_test_data(test_df, preprocessor, model_type, feature_list)
    if X_test_processed is None:
        print("❌ Failed to preprocess test data.")
        return
    
    # Generate predictions
    predictions = generate_predictions(model, X_test_processed, model_type)
    if predictions is None:
        print("❌ Failed to generate predictions.")
        return
    
    # Create submission file
    submission_df = create_submission_file(test_df, predictions)
    if submission_df is None:
        print("❌ Failed to create submission file.")
        return
    
    # Validate submission
    if validate_submission(submission_df):
        total_time = time.time() - start_time
        print(f"\n🎉 SUBMISSION CREATION SUCCESSFUL!")
        print(f"   File: submission.csv")
        print(f"   Samples: {len(submission_df):,}")
        print(f"   Model: {model_type.upper()}")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Ready for submission!")
    else:
        print("❌ Submission validation failed.")

if __name__ == "__main__":
    main() 