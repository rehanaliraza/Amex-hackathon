#!/usr/bin/env python3
"""
Quick AMEX Dataset Analysis
Get started with understanding the data structure and target variable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')

def load_sample_efficiently(filepath, sample_size=5000):
    """Load a representative sample from large parquet file"""
    # Use pyarrow to read efficiently
    table = pq.read_table(filepath)
    df = table.to_pandas()
    
    # Sample if dataset is larger than sample_size
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return df

def analyze_target_variable(df):
    """Analyze the target variable 'y'"""
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("=" * 50)
    
    if 'y' not in df.columns:
        print("No target variable 'y' found!")
        return None
    
    # Basic stats
    print(f"Target variable type: {df['y'].dtype}")
    print(f"Total samples: {len(df):,}")
    print(f"Missing values: {df['y'].isnull().sum()}")
    
    # Value counts
    value_counts = df['y'].value_counts().head(10)
    print(f"\nTop 10 target values:")
    for val, count in value_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {val}: {count:,} ({pct:.1f}%)")
    
    # Try to convert to numeric for analysis
    try:
        y_numeric = pd.to_numeric(df['y'], errors='coerce')
        if y_numeric.notna().sum() > len(df) * 0.8:  # If 80%+ are numeric
            print(f"\nNumeric target statistics:")
            print(y_numeric.describe())
            
            # Plot distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            y_numeric.hist(bins=30, alpha=0.7, color='skyblue')
            plt.title('Target Distribution (Numeric)')
            plt.xlabel('Target Value')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            value_counts.plot(kind='bar', alpha=0.7, color='lightcoral')
            plt.title('Target Value Counts')
            plt.xlabel('Target Value')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            return y_numeric
    except Exception as e:
        print(f"Could not convert to numeric: {e}")
    
    return df['y']

def categorize_features(df):
    """Categorize features by type based on data dictionary knowledge"""
    feature_categories = {
        'interest_scores': [f'f{i}' for i in range(1, 13)],  # f1-f12: Interest scores
        'web_behavior': [f'f{i}' for i in range(13, 85)],    # f13-f84: Web behavior
        'marketing_engagement': [f'f{i}' for i in range(85, 151)],  # f85-f150: Marketing
        'spending_patterns': [f'f{i}' for i in range(151, 199)],    # f151-f198: Spending
        'recent_activity': [f'f{i}' for i in range(199, 310)],      # f199-f309: Recent activity
        'offer_characteristics': [f'f{i}' for i in range(310, 367)], # f310-f366: Offer details
        'ids': [col for col in df.columns if col.startswith('id')],
        'target': ['y'] if 'y' in df.columns else []
    }
    
    return feature_categories

def quick_feature_analysis(df, categories):
    """Quick analysis of different feature categories"""
    print("\n📊 FEATURE CATEGORY ANALYSIS")
    print("=" * 50)
    
    for category, features in categories.items():
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            continue
            
        print(f"\n{category.upper().replace('_', ' ')} ({len(available_features)} features)")
        print("-" * 30)
        
        # Sample a few features from each category for analysis
        sample_features = available_features[:3]
        
        for feature in sample_features:
            try:
                # Try to get basic stats
                non_null_count = df[feature].notna().sum()
                null_pct = (df[feature].isnull().sum() / len(df)) * 100
                unique_count = df[feature].nunique()
                
                print(f"  {feature}: {non_null_count:,} non-null ({null_pct:.1f}% missing), {unique_count} unique")
                
                # Try numeric conversion
                if df[feature].dtype == 'object':
                    numeric_vals = pd.to_numeric(df[feature], errors='coerce')
                    if numeric_vals.notna().sum() > non_null_count * 0.8:
                        print(f"    → Numeric range: {numeric_vals.min():.2f} to {numeric_vals.max():.2f}")
                
            except Exception as e:
                print(f"  {feature}: Error analyzing - {str(e)[:50]}")

def analyze_data_relationships(df):
    """Look for obvious relationships in the data"""
    print("\n🔗 DATA RELATIONSHIPS")
    print("=" * 50)
    
    # Check ID columns for uniqueness
    id_cols = [col for col in df.columns if col.startswith('id')]
    print("ID Column Analysis:")
    for col in id_cols:
        unique_count = df[col].nunique()
        total_count = len(df)
        print(f"  {col}: {unique_count:,} unique values ({unique_count/total_count:.1%} unique)")
    
    # Look for highly correlated features (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        print(f"\nAnalyzing correlations among {len(numeric_cols)} numeric columns...")
        
        # Sample for correlation analysis to avoid memory issues
        sample_size = min(1000, len(df))
        df_sample = df[numeric_cols].sample(n=sample_size, random_state=42)
        
        # Calculate correlations
        corr_matrix = df_sample.corr()
        
        # Find high correlations (excluding self-correlations)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print("High correlations found:")
            for col1, col2, corr_val in high_corr_pairs[:10]:  # Show top 10
                print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
        else:
            print("No high correlations (>0.7) found in sample")

def main():
    """Main analysis function"""
    print("🚀 QUICK AMEX DATASET ANALYSIS")
    print("=" * 60)
    
    # Load training data sample
    print("Loading training data sample...")
    try:
        train_df = load_sample_efficiently('train_data.parquet', sample_size=5000)
        print(f"✅ Loaded {len(train_df):,} rows × {len(train_df.columns)} columns")
        
        # Analyze target variable
        target_data = analyze_target_variable(train_df)
        
        # Categorize features
        feature_categories = categorize_features(train_df)
        
        # Quick feature analysis
        quick_feature_analysis(train_df, feature_categories)
        
        # Analyze relationships
        analyze_data_relationships(train_df)
        
        print("\n" + "="*60)
        print("🎯 NEXT STEPS RECOMMENDATIONS:")
        print("="*60)
        print("1. Open the Jupyter notebook for interactive exploration")
        print("2. Focus on target variable distribution and class balance")
        print("3. Investigate feature engineering opportunities")
        print("4. Consider dimensionality reduction for 366 features")
        print("5. Explore additional datasets (events, transactions)")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Make sure you're in the correct directory with parquet files")

if __name__ == "__main__":
    main() 