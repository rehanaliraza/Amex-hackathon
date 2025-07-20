import pandas as pd
import numpy as np

def create_zero_submission():
    """
    Create a submission.csv file with all predictions set to zero.
    Uses the template file to get the correct structure and IDs.
    """
    
    # Read the template file
    print("Reading submission template...")
    template_df = pd.read_csv('685404e30cfdb_submission_template.csv')
    
    print(f"Template shape: {template_df.shape}")
    print(f"Template columns: {template_df.columns.tolist()}")
    
    # Create a copy of the template
    submission_df = template_df.copy()
    
    # Fill all predictions with zero
    submission_df['pred'] = 0.0
    
    # Save the submission file
    output_file = 'submission_zero.csv'
    submission_df.to_csv(output_file, index=False)
    
    print(f"Created submission file: {output_file}")
    print(f"Shape: {submission_df.shape}")
    print(f"Prediction range: {submission_df['pred'].min()} to {submission_df['pred'].max()}")
    print(f"Number of unique predictions: {submission_df['pred'].nunique()}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(submission_df.head())
    
    return output_file

if __name__ == "__main__":
    create_zero_submission() 