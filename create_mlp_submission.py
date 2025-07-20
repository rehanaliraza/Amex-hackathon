#!/usr/bin/env python3
"""
Create submission using MLP Neural Network Model
"""

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

def create_mlp_submission():
    """
    Create a submission.csv file using the MLP neural network model.
    """
    
    print("🚀 CREATING MLP NEURAL NETWORK SUBMISSION")
    print("=" * 60)
    
    try:
        # Load the trained model and preprocessors
        print("📊 Loading MLP model and preprocessors...")
        model = joblib.load('best_model_simple_mlp.pkl')
        scaler = joblib.load('scaler_simple_mlp.pkl')
        imputer = joblib.load('imputer_simple_mlp.pkl')
        
        # Load features
        with open('features_simple_mlp.txt', 'r') as f:
            features = [line.strip() for line in f.readlines()]
        
        print(f"✅ Loaded model with {len(features)} features")
        
        # Read the template file
        print("📊 Reading submission template...")
        template_df = pd.read_csv('685404e30cfdb_submission_template.csv')
        
        print(f"Template shape: {template_df.shape}")
        print(f"Template columns: {template_df.columns.tolist()}")
        
        # Load test data
        print("📊 Loading test data...")
        test_df = pd.read_parquet('test_data.parquet')
        print(f"Test data shape: {test_df.shape}")
        
        # Extract features from test data
        available_features = [f for f in features if f in test_df.columns]
        X_test = test_df[available_features].copy()
        
        print(f"✅ Using {len(available_features)} features from test data")
        
        # Convert to numeric
        for col in X_test.columns:
            if X_test[col].dtype == 'object':
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        
        # Handle missing values
        print("🔧 Preprocessing test data...")
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Scale features
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Make predictions
        print("🤖 Making predictions...")
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"✅ Generated {len(y_pred_proba)} predictions")
        print(f"📊 Prediction range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
        print(f"📊 Prediction mean: {y_pred_proba.mean():.4f}")
        
        # Create submission dataframe
        submission_df = template_df.copy()
        submission_df['pred'] = y_pred_proba
        
        # Save the submission file
        output_file = 'submission_mlp_neural_network.csv'
        submission_df.to_csv(output_file, index=False)
        
        print(f"✅ Created submission file: {output_file}")
        print(f"📊 Submission shape: {submission_df.shape}")
        print(f"📊 Prediction range: {submission_df['pred'].min():.4f} to {submission_df['pred'].max():.4f}")
        
        # Show first few rows
        print("\n📋 First 5 rows of submission:")
        print(submission_df.head())
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error creating submission: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("🚀 MLP NEURAL NETWORK SUBMISSION CREATION")
    print("=" * 80)
    
    output_file = create_mlp_submission()
    
    if output_file:
        print(f"\n🎯 SUBMISSION CREATION COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully created MLP neural network submission")
        print(f"📁 Output file: {output_file}")
        print(f"📊 File size: {pd.read_csv(output_file).shape}")
        
        # Check file size
        import os
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"💾 File size: {file_size:.1f} MB")
        
    else:
        print("❌ Submission creation failed")

if __name__ == "__main__":
    main() 