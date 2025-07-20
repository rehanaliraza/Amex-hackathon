#!/usr/bin/env python3
"""
Create Submission File using Balanced Neural Network
Standalone script to generate predictions and create submission file
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import time
import gc

warnings.filterwarnings('ignore')

def load_model_and_preprocessors():
    """Load the trained model and preprocessors"""
    print("🔄 Loading model and preprocessors...")
    
    try:
        # Load model and preprocessors
        model = joblib.load('best_model_balanced_nn.pkl')
        scaler = joblib.load('scaler_balanced_nn.pkl')
        imputer = joblib.load('imputer_balanced_nn.pkl')
        
        # Load feature list
        with open('features_balanced_nn.txt', 'r') as f:
            important_features = [line.strip() for line in f.readlines()]
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Using {len(important_features)} features")
        
        return model, scaler, imputer, important_features
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please run train_balanced_neural_network.py first")
        return None, None, None, None

def preprocess_test_data(test_df, important_features, imputer):
    """Preprocess test data for prediction"""
    print("🔧 Preprocessing test data...")
    
    # Extract available features
    available_features = [f for f in important_features if f in test_df.columns]
    X_test = test_df[available_features].copy()
    
    print(f"   Requested features: {len(important_features)}")
    print(f"   Available features: {len(available_features)}")
    print(f"   Missing features: {len(important_features) - len(available_features)}")
    
    # Convert to numeric
    print("   Converting object columns to numeric...")
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
    
    print(f"   Missing values after imputation: {X_test_imputed.isnull().sum().sum()}")
    
    return X_test_imputed, available_features

def create_predictions(model, scaler, X_test_imputed):
    """Create predictions using the trained model"""
    print("🧠 Making predictions...")
    
    # Scale features
    print("   Scaling features...")
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Make probability predictions (0.0 to 1.0)
    print("   Generating probability predictions...")
    predictions = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
    print(f"   Mean prediction: {predictions.mean():.4f}")
    
    return predictions

def create_submission_file(test_df, predictions, filename='submission_balanced_nn.csv'):
    """Create and save submission file"""
    print(f"📝 Creating submission file: {filename}")
    
    # Create submission dataframe with required columns
    submission_df = pd.DataFrame({
        'id1': test_df['id1'],
        'id2': test_df['id2'], 
        'id3': test_df['id3'],
        'id5': test_df['id5'],
        'pred': predictions
    })
    
    # Save submission
    submission_df.to_csv(filename, index=False)
    
    print(f"✅ Submission file saved: {filename}")
    print(f"   File size: {len(submission_df):,} predictions")
    print(f"   Columns: {submission_df.columns.tolist()}")
    
    # Show prediction distribution
    print(f"\n📊 Prediction Distribution:")
    print(f"   Min: {predictions.min():.4f}")
    print(f"   Max: {predictions.max():.4f}")
    print(f"   Mean: {predictions.mean():.4f}")
    print(f"   Std: {predictions.std():.4f}")
    
    # Show quantiles for probability distribution
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print(f"   Quantiles:")
    for q in quantiles:
        value = np.quantile(predictions, q)
        print(f"     {q*100:3.0f}%: {value:.4f}")
    
    # Show prediction ranges
    low_pred = (predictions < 0.1).sum()
    high_pred = (predictions > 0.9).sum()
    mid_pred = ((predictions >= 0.1) & (predictions <= 0.9)).sum()
    print(f"   Prediction ranges:")
    print(f"     < 0.1: {low_pred:,} ({low_pred/len(predictions)*100:.1f}%)")
    print(f"     0.1-0.9: {mid_pred:,} ({mid_pred/len(predictions)*100:.1f}%)")
    print(f"     > 0.9: {high_pred:,} ({high_pred/len(predictions)*100:.1f}%)")
    
    return filename

def main():
    """Main function to create submission"""
    print("🚀 CREATING BALANCED NEURAL NETWORK SUBMISSION")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load model and preprocessors
        model, scaler, imputer, important_features = load_model_and_preprocessors()
        
        if model is None:
            return False
        
        # Load test data
        print("\n📊 Loading test data...")
        test_df = pd.read_parquet('test_data.parquet')
        print(f"✅ Test data loaded: {len(test_df):,} samples")
        
        # Preprocess test data
        X_test_imputed, available_features = preprocess_test_data(
            test_df, important_features, imputer
        )
        
        # Create predictions
        predictions = create_predictions(model, scaler, X_test_imputed)
        
        # Create submission file
        submission_filename = create_submission_file(test_df, predictions)
        
        # Clean up
        del test_df, X_test_imputed
        gc.collect()
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 SUBMISSION CREATION COMPLETE!")
        print("=" * 50)
        print(f"✅ Successfully created submission file")
        print(f"📊 Features used: {len(available_features)}")
        print(f"📊 Test samples: {len(predictions):,}")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"📝 Submission file: {submission_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Submission creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 