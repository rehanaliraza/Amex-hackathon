#!/usr/bin/env python3
"""
Comprehensive Feature Importance Analysis for AMEX Dataset
Analyzes existing models to identify most important features for full dataset training
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import warnings
import time
import psutil
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

def load_existing_models():
    """Load all existing trained models and their feature importances"""
    print("🔍 LOADING EXISTING MODELS FOR FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    models_data = []
    
    # Check for available models
    model_files = [
        ('Auto Random Forest', 'best_model_auto_rf_balanced.pkl', 'data_preprocessor_auto_rf_balanced.pkl'),
        ('Simple Random Forest', 'best_model_simple_rf_balanced.pkl', 'data_preprocessor_simple_rf.pkl'),
        ('10% Gradient Boosting', 'best_model_10pct_gradient_boosting.pkl', 'data_preprocessor_10pct.pkl'),
    ]
    
    for model_name, model_file, preprocessor_file in model_files:
        if Path(model_file).exists() and Path(preprocessor_file).exists():
            try:
                print(f"📊 Loading {model_name}...")
                model = joblib.load(model_file)
                preprocessor = joblib.load(preprocessor_file)
                
                if hasattr(model, 'feature_importances_'):
                    # Get feature names from preprocessor
                    if hasattr(preprocessor, 'feature_names_in_'):
                        feature_names = preprocessor.feature_names_in_
                    else:
                        # Fallback: assume feature names are f1, f2, etc.
                        feature_names = [f'f{i}' for i in range(1, len(model.feature_importances_) + 1)]
                    
                    models_data.append({
                        'name': model_name,
                        'model': model,
                        'feature_names': feature_names,
                        'importances': model.feature_importances_
                    })
                    print(f"   ✅ {model_name}: {len(model.feature_importances_)} features")
                else:
                    print(f"   ⚠️ {model_name}: No feature importance available")
                    
            except Exception as e:
                print(f"   ❌ {model_name}: Error loading - {e}")
    
    print(f"\n✅ Loaded {len(models_data)} models with feature importance")
    return models_data

def analyze_feature_importance_consensus(models_data):
    """Analyze feature importance consensus across multiple models"""
    print("\n🎯 FEATURE IMPORTANCE CONSENSUS ANALYSIS")
    print("=" * 60)
    
    if not models_data:
        print("❌ No models available for analysis")
        return None
    
    # Create a combined importance dataframe
    all_importances = []
    
    for model_data in models_data:
        importance_df = pd.DataFrame({
            'feature': model_data['feature_names'],
            'importance': model_data['importances'],
            'model': model_data['name']
        })
        all_importances.append(importance_df)
    
    # Combine all importances
    combined_df = pd.concat(all_importances, ignore_index=True)
    
    # Calculate consensus metrics
    consensus_stats = combined_df.groupby('feature')['importance'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Calculate consensus score (mean importance with stability bonus)
    consensus_stats['consensus_score'] = (
        consensus_stats['mean'] * 
        (1 + 1 / (1 + consensus_stats['std']))  # Stability bonus
    )
    
    # Sort by consensus score
    consensus_stats = consensus_stats.sort_values('consensus_score', ascending=False)
    
    print(f"📊 Feature Importance Consensus (Top 25):")
    print(f"{'Rank':<4} {'Feature':<10} {'Mean Imp':<10} {'Std':<8} {'Models':<7} {'Score':<10}")
    print("-" * 60)
    
    for i, (_, row) in enumerate(consensus_stats.head(25).iterrows()):
        print(f"{i+1:<4} {row['feature']:<10} {row['mean']:<10.4f} {row['std']:<8.4f} {int(row['count']):<7} {row['consensus_score']:<10.4f}")
    
    return consensus_stats

def categorize_features_by_importance(consensus_stats):
    """Categorize features into importance tiers"""
    print("\n📋 FEATURE CATEGORIZATION BY IMPORTANCE")
    print("=" * 50)
    
    # Calculate cumulative importance
    total_importance = consensus_stats['consensus_score'].sum()
    consensus_stats['cumulative_pct'] = (
        consensus_stats['consensus_score'].cumsum() / total_importance * 100
    )
    
    # Define importance tiers
    tier1_features = consensus_stats[consensus_stats['cumulative_pct'] <= 50]['feature'].tolist()
    tier2_features = consensus_stats[
        (consensus_stats['cumulative_pct'] > 50) & 
        (consensus_stats['cumulative_pct'] <= 80)
    ]['feature'].tolist()
    tier3_features = consensus_stats[
        (consensus_stats['cumulative_pct'] > 80) & 
        (consensus_stats['cumulative_pct'] <= 95)
    ]['feature'].tolist()
    
    print(f"🥇 Tier 1 (Top 50% importance): {len(tier1_features)} features")
    print(f"   Features: {tier1_features[:10]}{'...' if len(tier1_features) > 10 else ''}")
    
    print(f"🥈 Tier 2 (50-80% importance): {len(tier2_features)} features")
    print(f"   Features: {tier2_features[:10]}{'...' if len(tier2_features) > 10 else ''}")
    
    print(f"🥉 Tier 3 (80-95% importance): {len(tier3_features)} features")
    print(f"   Features: {tier3_features[:10]}{'...' if len(tier3_features) > 10 else ''}")
    
    remaining_features = consensus_stats[consensus_stats['cumulative_pct'] > 95]['feature'].tolist()
    print(f"📉 Remaining (95%+ importance): {len(remaining_features)} features")
    
    return {
        'tier1': tier1_features,
        'tier2': tier2_features,
        'tier3': tier3_features,
        'remaining': remaining_features
    }

def validate_feature_selection(feature_tiers, sample_size=50000):
    """Validate feature selection by training models with different feature sets"""
    print("\n🧪 VALIDATING FEATURE SELECTION")
    print("=" * 50)
    
    # Load a sample of data for validation
    print(f"📊 Loading {sample_size:,} samples for validation...")
    try:
        train_df = pd.read_parquet('train_data.parquet')
        if sample_size < len(train_df):
            train_df = train_df.sample(n=sample_size, random_state=42)
        
        # Preprocess
        feature_cols = [col for col in train_df.columns if col.startswith('f')]
        X = train_df[feature_cols].copy()
        y = pd.to_numeric(train_df['y'], errors='coerce')
        
        # Convert to numeric and clean
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Remove empty columns
        valid_cols = [col for col in X.columns if X[col].notna().sum() > 0]
        X = X[valid_cols]
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"✅ Data prepared: {len(X_imputed)} samples, {len(X_imputed.columns)} features")
        
        # Test different feature sets
        feature_sets = {
            'All Features': X_imputed.columns.tolist(),
            'Tier 1 Only': [f for f in feature_tiers['tier1'] if f in X_imputed.columns],
            'Tier 1+2': [f for f in feature_tiers['tier1'] + feature_tiers['tier2'] if f in X_imputed.columns],
            'Tier 1+2+3': [f for f in feature_tiers['tier1'] + feature_tiers['tier2'] + feature_tiers['tier3'] if f in X_imputed.columns],
            'Top 50': [f for f in feature_tiers['tier1'][:50] if f in X_imputed.columns],
            'Top 100': [f for f in (feature_tiers['tier1'] + feature_tiers['tier2'])[:100] if f in X_imputed.columns],
        }
        
        results = {}
        
        for set_name, features in feature_sets.items():
            if len(features) == 0:
                continue
                
            print(f"\n🔄 Testing {set_name} ({len(features)} features)...")
            
            # Select features
            X_subset = X_imputed[features]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_subset, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest
            start_time = time.time()
            rf = RandomForestClassifier(
                n_estimators=50,  # Reduced for speed
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                max_depth=10
            )
            
            rf.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = rf.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            training_time = time.time() - start_time
            
            results[set_name] = {
                'n_features': len(features),
                'auc_score': auc_score,
                'training_time': training_time,
                'memory_reduction': 1 - (len(features) / len(X_imputed.columns))
            }
            
            print(f"   ✅ AUC: {auc_score:.4f}, Time: {training_time:.2f}s")
            
            # Clear memory
            del X_train, X_val, X_subset
            gc.collect()
        
        # Display results
        print(f"\n📊 FEATURE SELECTION VALIDATION RESULTS")
        print("=" * 70)
        print(f"{'Feature Set':<15} {'Features':<9} {'AUC':<8} {'Time(s)':<8} {'Memory↓':<8}")
        print("-" * 70)
        
        for set_name, result in results.items():
            print(f"{set_name:<15} {result['n_features']:<9} {result['auc_score']:<8.4f} {result['training_time']:<8.2f} {result['memory_reduction']:<8.1%}")
        
        return results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

def generate_feature_recommendations(consensus_stats, feature_tiers, validation_results=None):
    """Generate feature selection recommendations"""
    print("\n💡 FEATURE SELECTION RECOMMENDATIONS")
    print("=" * 60)
    
    # Memory-based recommendations
    memory_gb = psutil.virtual_memory().available / (1024**3)
    
    if memory_gb >= 6:
        recommended_features = feature_tiers['tier1'] + feature_tiers['tier2']
        recommendation = "High Memory"
        rationale = "Use Tier 1 + Tier 2 features for best performance"
    elif memory_gb >= 4:
        recommended_features = feature_tiers['tier1'] + feature_tiers['tier2'][:len(feature_tiers['tier2'])//2]
        recommendation = "Medium Memory"
        rationale = "Use Tier 1 + half of Tier 2 features"
    else:
        recommended_features = feature_tiers['tier1'][:100]
        recommendation = "Low Memory"
        rationale = "Use top 100 Tier 1 features only"
    
    print(f"🎯 RECOMMENDATION: {recommendation}")
    print(f"   Available memory: {memory_gb:.1f} GB")
    print(f"   Recommended features: {len(recommended_features)}")
    print(f"   Rationale: {rationale}")
    
    # Performance vs Memory trade-off
    print(f"\n⚖️ PERFORMANCE VS MEMORY TRADE-OFFS:")
    
    trade_offs = [
        ("Conservative", feature_tiers['tier1'][:50], "Fastest training, lowest memory"),
        ("Balanced", feature_tiers['tier1'][:100], "Good balance of performance and speed"),
        ("Aggressive", feature_tiers['tier1'] + feature_tiers['tier2'][:50], "Better performance, more memory"),
        ("Maximum", feature_tiers['tier1'] + feature_tiers['tier2'], "Best performance, highest memory")
    ]
    
    for strategy, features, description in trade_offs:
        memory_reduction = 1 - (len(features) / len(consensus_stats))
        print(f"   {strategy:<12}: {len(features):>3} features ({memory_reduction:>5.1%} memory reduction) - {description}")
    
    # Validation-based recommendations
    if validation_results:
        print(f"\n📈 VALIDATION-BASED INSIGHTS:")
        
        # Find best performance per feature count
        sorted_results = sorted(validation_results.items(), key=lambda x: x[1]['auc_score'], reverse=True)
        best_overall = sorted_results[0]
        
        # Find best efficiency (AUC per feature)
        efficiency_results = [(name, result['auc_score'] / result['n_features']) 
                             for name, result in validation_results.items()]
        efficiency_results.sort(key=lambda x: x[1], reverse=True)
        best_efficiency = efficiency_results[0]
        
        print(f"   Best Performance: {best_overall[0]} (AUC: {best_overall[1]['auc_score']:.4f})")
        print(f"   Best Efficiency: {best_efficiency[0]} (AUC/Feature: {best_efficiency[1]:.6f})")
    
    return {
        'recommended_features': recommended_features,
        'strategy': recommendation,
        'alternatives': {name: features for name, features, _ in trade_offs}
    }

def save_feature_selections(consensus_stats, feature_tiers, recommendations):
    """Save feature selections to files for use in training"""
    print("\n💾 SAVING FEATURE SELECTIONS")
    print("=" * 40)
    
    # Save full feature importance analysis
    consensus_stats.to_csv('feature_importance_analysis.csv', index=False)
    print(f"✅ Full analysis saved: feature_importance_analysis.csv")
    
    # Save different feature sets
    feature_sets = {
        'top_50_features.txt': recommendations['alternatives']['Conservative'],
        'top_100_features.txt': recommendations['alternatives']['Balanced'],
        'recommended_features.txt': recommendations['recommended_features'],
        'tier1_features.txt': feature_tiers['tier1'],
        'tier1_tier2_features.txt': feature_tiers['tier1'] + feature_tiers['tier2'],
    }
    
    for filename, features in feature_sets.items():
        with open(filename, 'w') as f:
            for feature in features:
                f.write(f"{feature}\n")
        print(f"✅ Saved {len(features)} features: {filename}")
    
    print(f"\n📋 Feature selection files ready for training scripts!")

def create_feature_importance_visualization(consensus_stats, feature_tiers):
    """Create visualizations of feature importance"""
    print("\n📊 CREATING FEATURE IMPORTANCE VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top 20 features bar plot
        top_20 = consensus_stats.head(20)
        axes[0, 0].barh(range(len(top_20)), top_20['consensus_score'])
        axes[0, 0].set_yticks(range(len(top_20)))
        axes[0, 0].set_yticklabels(top_20['feature'])
        axes[0, 0].set_xlabel('Consensus Score')
        axes[0, 0].set_title('Top 20 Most Important Features')
        axes[0, 0].invert_yaxis()
        
        # 2. Feature importance distribution
        axes[0, 1].hist(consensus_stats['consensus_score'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Consensus Score')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Feature Importance Distribution')
        axes[0, 1].axvline(consensus_stats['consensus_score'].median(), color='red', linestyle='--', label='Median')
        axes[0, 1].legend()
        
        # 3. Cumulative importance
        axes[1, 0].plot(range(len(consensus_stats)), consensus_stats['cumulative_pct'])
        axes[1, 0].set_xlabel('Feature Rank')
        axes[1, 0].set_ylabel('Cumulative Importance (%)')
        axes[1, 0].set_title('Cumulative Feature Importance')
        axes[1, 0].axhline(50, color='red', linestyle='--', label='50%')
        axes[1, 0].axhline(80, color='orange', linestyle='--', label='80%')
        axes[1, 0].axhline(95, color='yellow', linestyle='--', label='95%')
        axes[1, 0].legend()
        
        # 4. Feature tiers pie chart
        tier_sizes = [len(feature_tiers['tier1']), len(feature_tiers['tier2']), 
                     len(feature_tiers['tier3']), len(feature_tiers['remaining'])]
        tier_labels = ['Tier 1 (50%)', 'Tier 2 (30%)', 'Tier 3 (15%)', 'Remaining (5%)']
        colors = ['gold', 'silver', '#CD7F32', 'lightgray']
        
        axes[1, 1].pie(tier_sizes, labels=tier_labels, colors=colors, autopct='%1.1f%%')
        axes[1, 1].set_title('Feature Distribution by Importance Tiers')
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: feature_importance_analysis.png")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")

def main():
    """Main feature importance analysis function"""
    print("🚀 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Load existing models
        models_data = load_existing_models()
        
        if not models_data:
            print("❌ No models found for analysis. Please train models first.")
            return False
        
        # Analyze feature importance consensus
        consensus_stats = analyze_feature_importance_consensus(models_data)
        
        if consensus_stats is None:
            return False
        
        # Categorize features by importance
        feature_tiers = categorize_features_by_importance(consensus_stats)
        
        # Validate feature selection
        validation_results = validate_feature_selection(feature_tiers)
        
        # Generate recommendations
        recommendations = generate_feature_recommendations(
            consensus_stats, feature_tiers, validation_results
        )
        
        # Save feature selections
        save_feature_selections(consensus_stats, feature_tiers, recommendations)
        
        # Create visualizations
        create_feature_importance_visualization(consensus_stats, feature_tiers)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"✅ Analyzed {len(models_data)} models")
        print(f"✅ Processed {len(consensus_stats)} features")
        print(f"✅ Recommended {len(recommendations['recommended_features'])} features")
        print(f"✅ Total time: {total_time:.2f} seconds")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Use recommended features for full dataset training")
        print(f"   2. Run: python train_full_gb_optimized.py")
        print(f"   3. Compare performance with full feature set")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 