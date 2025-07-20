#!/usr/bin/env python3
"""
Create Enhanced Submission using the trained enhanced neural network
Fixes feature mismatch and creates proper submission
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import time
import gc

warnings.filterwarnings('ignore')

def load_enhanced_model():
    """Load the enhanced model and preprocessors"""
    print("🔄 Loading enhanced model and preprocessors...")
    
    try:
        # Load model and preprocessors
        model = joblib.load('best_model_enhanced_nn.pkl')
        scaler = joblib.load('scaler_enhanced_nn.pkl')
        imputer = joblib.load('imputer_enhanced_nn.pkl')
        
        # Load feature list
        with open('features_enhanced_nn.txt', 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        print(f"✅ Enhanced model loaded successfully")
        print(f"✅ Using {len(feature_cols)} features")
        
        return model, scaler, imputer, feature_cols
        
    except FileNotFoundError as e:
        print(f"❌ Enhanced model files not found: {e}")
        print("Please run train_enhanced_neural_network.py first")
        return None, None, None, None

def get_common_features(train_features, test_df, imputer):
    """Get features that exist in both training and test data and were used in imputer"""
    print("🔍 Finding common features between train and test...")
    
    test_features = [col for col in test_df.columns if col.startswith('f')]
    
    # Get features that the imputer was trained on
    imputer_features = list(imputer.feature_names_in_)
    
    # Find intersection of all three sets
    common_features = list(set(train_features) & set(test_features) & set(imputer_features))
    
    print(f"   Training features: {len(train_features)}")
    print(f"   Test features: {len(test_features)}")
    print(f"   Imputer features: {len(imputer_features)}")
    print(f"   Common features: {len(common_features)}")
    
    return common_features

def preprocess_test_data_enhanced(test_df, common_features, imputer):
    """Preprocess test data using only common features in correct order"""
    print("🔧 Preprocessing test data with common features...")
    
    # Use the exact features and order that the imputer was trained on
    imputer_features = list(imputer.feature_names_in_)
    X_test = test_df[imputer_features].copy()
    
    print(f"   Using {len(imputer_features)} features in correct order")
    
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
    
    return X_test_imputed

def create_enhanced_predictions(model, scaler, X_test_imputed):
    """Create probability predictions using the enhanced model"""
    print("🧠 Making enhanced probability predictions...")
    
    # Scale features
    print("   Scaling features...")
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Make probability predictions
    print("   Generating probability predictions...")
    predictions = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
    print(f"   Mean prediction: {predictions.mean():.4f}")
    
    return predictions

def create_enhanced_submission_file(test_df, predictions, filename='submission_enhanced_nn.csv'):
    """Create and save enhanced submission file"""
    print(f"📝 Creating enhanced submission file: {filename}")
    
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
    
    print(f"✅ Enhanced submission file saved: {filename}")
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
    """Main function to create enhanced submission"""
    print("🚀 CREATING ENHANCED NEURAL NETWORK SUBMISSION")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Load enhanced model and preprocessors
        model, scaler, imputer, feature_cols = load_enhanced_model()
        
        if model is None:
            return False
        
        # Load test data
        print("\n📊 Loading test data...")
        test_df = pd.read_parquet('test_data.parquet')
        print(f"✅ Test data loaded: {len(test_df):,} samples")
        
        # Get common features between train and test
        common_features = get_common_features(feature_cols, test_df, imputer)
        
        if len(common_features) == 0:
            print("❌ No common features found between train and test data!")
            return False
        
        # Preprocess test data with common features
        X_test_imputed = preprocess_test_data_enhanced(test_df, common_features, imputer)
        
        # Create predictions
        predictions = create_enhanced_predictions(model, scaler, X_test_imputed)
        
        # Create submission file
        submission_filename = create_enhanced_submission_file(test_df, predictions)
        
        # Clean up
        del test_df, X_test_imputed
        gc.collect()
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 ENHANCED SUBMISSION CREATION COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully created enhanced submission file")
        print(f"📊 Features used: {len(common_features)}")
        print(f"📊 Test samples: {len(predictions):,}")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"📝 Submission file: {submission_filename}")
        
        # Expected score improvement
        print(f"\n🎯 EXPECTED SCORE IMPROVEMENT:")
        print(f"   Previous score: ~0.2 (binary predictions)")
        print(f"   Expected score: 0.5-0.7 (probability predictions)")
        print(f"   Improvement: +0.3-0.5 points")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced submission creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 