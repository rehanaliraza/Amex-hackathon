#!/usr/bin/env python3
"""
Efficient Parquet File Explorer
Explores large parquet files without loading them entirely into memory
"""

import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def explore_parquet_metadata(filepath):
    """Get metadata without loading the full file"""
    print(f"\n{'='*60}")
    print(f"EXPLORING: {filepath}")
    print(f"File size: {get_file_size_mb(filepath):.1f} MB")
    print(f"{'='*60}")
    
    try:
        # Get parquet metadata
        parquet_file = pq.ParquetFile(filepath)
        
        # Basic info
        print(f"Number of rows: {parquet_file.metadata.num_rows:,}")
        print(f"Number of columns: {parquet_file.schema_arrow.names.__len__()}")
        print(f"Number of row groups: {parquet_file.metadata.num_row_groups}")
        
        # Schema info
        print("\nCOLUMN SCHEMA:")
        schema = parquet_file.schema_arrow
        for i, field in enumerate(schema):
            print(f"  {i+1:2d}. {field.name:25s} | {field.type}")
        
        return parquet_file
        
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None

def sample_data(filepath, n_rows=5):
    """Get a small sample of the data"""
    print(f"\nSAMPLE DATA (first {n_rows} rows):")
    print("-" * 60)
    
    try:
        # Use pyarrow to read limited rows then convert to pandas
        parquet_file = pq.ParquetFile(filepath)
        
        # Read first row group and limit rows
        table = parquet_file.read_row_group(0)
        df_sample = table.to_pandas().head(n_rows)
        
        print(df_sample.to_string())
        
        print(f"\nDATA TYPES:")
        print(df_sample.dtypes.to_string())
        
        return df_sample
        
    except Exception as e:
        print(f"Error sampling data: {e}")
        return None

def get_memory_usage(filepath, sample_size=1000):
    """Estimate memory usage per row"""
    try:
        # Use pyarrow to read limited rows
        parquet_file = pq.ParquetFile(filepath)
        table = parquet_file.read_row_group(0)
        sample = table.to_pandas().head(sample_size)
        
        memory_per_row = sample.memory_usage(deep=True).sum() / len(sample)
        
        total_rows = parquet_file.metadata.num_rows
        estimated_full_memory_gb = (memory_per_row * total_rows) / (1024**3)
        
        print(f"\nMEMORY ESTIMATES:")
        print(f"Memory per row: {memory_per_row:.0f} bytes")
        print(f"Estimated full load: {estimated_full_memory_gb:.2f} GB")
        
        return estimated_full_memory_gb
        
    except Exception as e:
        print(f"Error estimating memory: {e}")
        return None

def main():
    """Main exploration function"""
    print("PARQUET FILE EXPLORER")
    print("=" * 60)
    
    # Find all parquet files
    parquet_files = list(Path('.').glob('*.parquet'))
    
    if not parquet_files:
        print("No parquet files found in current directory")
        return
    
    print(f"Found {len(parquet_files)} parquet files:")
    for i, file in enumerate(parquet_files):
        size_mb = get_file_size_mb(file)
        print(f"  {i+1}. {file.name} ({size_mb:.1f} MB)")
    
    # Explore each file
    for file in parquet_files:
        explore_parquet_metadata(file)
        sample_data(file, n_rows=3)
        get_memory_usage(file)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 