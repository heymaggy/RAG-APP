"""
Fast FAERS Drug Processing - Extract unique drugs first, then map
This reduces API calls by ~99% for typical FAERS files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from FAERSRxNormMapper import FAERSRxNormMapper  # Import from the main mapper file

# Step 1: Extract unique drugs from FAERS
def extract_unique_faers_drugs(file_path: str, chunksize: int = 100000):
    """
    Extract unique drugs from FAERS drug file efficiently
    """
    print("="*60)
    print("EXTRACTING UNIQUE DRUGS FROM FAERS FILE")
    print("="*60)
    
    start_time = time.time()
    unique_drugs = set()
    total_rows = 0
    chunk_count = 0
    
    # Track additional statistics
    drug_stats = {
        'total_null': 0,
        'total_entries': 0
    }
    
    print(f"\nProcessing file: {file_path}")
    print(f"Reading in chunks of {chunksize:,} rows...")
    
    # Read file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunksize, 
                            usecols=['primaryid', 'drugname'], 
                            dtype={'primaryid': str},
                            low_memory=False):
        chunk_count += 1
        total_rows += len(chunk)
        
        # Count nulls
        drug_stats['total_null'] += chunk['drugname'].isnull().sum()
        drug_stats['total_entries'] += len(chunk)
        
        # Extract unique drugs from this chunk
        chunk_drugs = chunk['drugname'].dropna().str.strip().unique()
        unique_drugs.update(chunk_drugs)
        
        # Progress update every 10 chunks
        if chunk_count % 10 == 0:
            elapsed = time.time() - start_time
            rate = total_rows / elapsed
            print(f"  Chunk {chunk_count}: {total_rows:,} rows, "
                  f"{len(unique_drugs):,} unique drugs "
                  f"({rate:.0f} rows/sec)")
    
    # Final statistics
    elapsed_time = time.time() - start_time
    print(f"\nExtraction Complete!")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total unique drugs: {len(unique_drugs):,}")
    print(f"  Null drug names: {drug_stats['total_null']:,}")
    print(f"  Processing time: {elapsed_time/60:.1f} minutes")
    print(f"  Processing rate: {total_rows/elapsed_time:.0f} rows/second")
    
    # Convert to DataFrame and sort by drug name
    unique_df = pd.DataFrame({'drugname': sorted(list(unique_drugs))})
    
    # Save to CSV
    output_file = 'faers_unique_drugs.csv'
    unique_df.to_csv(output_file, index=False)
    print(f"\nSaved unique drugs to: {output_file}")
    
    return unique_df, drug_stats

# Step 2: Map unique drugs to RxNorm
def map_unique_drugs_batch(unique_drugs_df: pd.DataFrame, 
                          start_from: int = 0,
                          batch_size: int = 1000):
    """
    Map unique drugs to RxNorm with progress tracking and ability to resume
    """
    print("\n" + "="*60)
    print("MAPPING UNIQUE DRUGS TO RXNORM")
    print("="*60)
    
    # Initialize mapper
    mapper = FAERSRxNormMapper(cache_db="faers_rxnorm_cache.db")
    
    # Add result columns
    if 'rxnorm_name' not in unique_drugs_df.columns:
        unique_drugs_df['rxnorm_name'] = None
        unique_drugs_df['rxcui'] = None
        unique_drugs_df['mapping_score'] = None
        unique_drugs_df['mapping_method'] = None
    
    total_drugs = len(unique_drugs_df)
    print(f"\nTotal drugs to map: {total_drugs:,}")
    print(f"Starting from index: {start_from}")
    
    # Time estimation
    drugs_to_process = total_drugs - start_from
    estimated_hours = drugs_to_process / (15 * 3600)  # 15 requests/sec
    print(f"Estimated time: {estimated_hours:.1f} hours")
    
    start_time = time.time()
    last_save_time = time.time()
    
    # Process in batches
    for i in range(start_from, total_drugs, batch_size):
        batch_end = min(i + batch_size, total_drugs)
        batch_num = (i // batch_size) + 1
        total_batches = (total_drugs // batch_size) + 1
        
        print(f"\nBatch {batch_num}/{total_batches} (drugs {i+1}-{batch_end})...")
        
        # Process batch
        for idx in range(i, batch_end):
            drug = unique_drugs_df.iloc[idx]['drugname']
            
            # Skip if already mapped
            if pd.notna(unique_drugs_df.iloc[idx]['rxcui']):
                continue
            
            # Map drug
            result = mapper.get_standardized_drug_name(drug)
            
            if result:
                unique_drugs_df.loc[idx, 'rxnorm_name'] = result.get('standard_name')
                unique_drugs_df.loc[idx, 'rxcui'] = result.get('rxcui')
                unique_drugs_df.loc[idx, 'mapping_score'] = result.get('score')
                unique_drugs_df.loc[idx, 'mapping_method'] = result.get('mapping_method')
        
        # Progress statistics
        elapsed = time.time() - start_time
        processed = batch_end - start_from
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = total_drugs - batch_end
        eta = remaining / rate if rate > 0 else 0
        
        success_count = unique_drugs_df.iloc[:batch_end]['rxcui'].notna().sum()
        success_rate = success_count / batch_end * 100
        
        print(f"  Progress: {batch_end}/{total_drugs} ({batch_end/total_drugs*100:.1f}%)")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Processing rate: {rate:.1f} drugs/second")
        print(f"  ETA: {eta/3600:.1f} hours")
        
        # Save progress every 5 minutes
        if time.time() - last_save_time > 300:  # 5 minutes
            save_file = f'faers_drugs_mapped_checkpoint_{batch_end}.csv'
            unique_drugs_df.to_csv(save_file, index=False)
            print(f"  Checkpoint saved: {save_file}")
            last_save_time = time.time()
    
    # Final save
    final_file = 'faers_unique_drugs_mapped_final.csv'
    unique_drugs_df.to_csv(final_file, index=False)
    print(f"\nMapping complete! Saved to: {final_file}")
    
    # Final statistics
    total_time = time.time() - start_time
    success_count = unique_drugs_df['rxcui'].notna().sum()
    
    print(f"\nFinal Statistics:")
    print(f"  Total processing time: {total_time/3600:.1f} hours")
    print(f"  Drugs successfully mapped: {success_count:,} ({success_count/total_drugs*100:.1f}%)")
    print(f"  Average rate: {total_drugs/total_time:.1f} drugs/second")
    
    return unique_drugs_df

# Step 3: Apply mappings back to full file
def apply_mappings_to_faers(original_file: str, mapping_df: pd.DataFrame,
                           output_file: str = 'faers_drug_mapped.csv',
                           chunksize: int = 100000):
    """
    Apply RxNorm mappings back to the full FAERS file
    """
    print("\n" + "="*60)
    print("APPLYING MAPPINGS TO FULL FAERS FILE")
    print("="*60)
    
    # Create mapping dictionary for fast lookup
    mapping_dict = mapping_df.set_index('drugname')[
        ['rxnorm_name', 'rxcui', 'mapping_score', 'mapping_method']
    ].to_dict('index')
    
    print(f"Created mapping dictionary with {len(mapping_dict):,} entries")
    
    start_time = time.time()
    first_chunk = True
    total_rows = 0
    mapped_count = 0
    
    # Process file in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(original_file, chunksize=chunksize,
                                                  low_memory=False)):
        # Apply mappings
        chunk['rxnorm_name'] = chunk['drugname'].map(
            lambda x: mapping_dict.get(x.strip() if pd.notna(x) else '', {}).get('rxnorm_name')
        )
        chunk['rxcui'] = chunk['drugname'].map(
            lambda x: mapping_dict.get(x.strip() if pd.notna(x) else '', {}).get('rxcui')
        )
        chunk['mapping_score'] = chunk['drugname'].map(
            lambda x: mapping_dict.get(x.strip() if pd.notna(x) else '', {}).get('mapping_score')
        )
        chunk['mapping_method'] = chunk['drugname'].map(
            lambda x: mapping_dict.get(x.strip() if pd.notna(x) else '', {}).get('mapping_method')
        )
        
        # Count mapped entries
        mapped_count += chunk['rxcui'].notna().sum()
        
        # Write to output
        if first_chunk:
            chunk.to_csv(output_file, index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_file, mode='a', header=False, index=False)
        
        total_rows += len(chunk)
        
        # Progress update
        if (chunk_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = total_rows / elapsed
            print(f"  Processed {total_rows:,} rows ({rate:.0f} rows/sec)")
    
    # Final statistics
    total_time = time.time() - start_time
    mapping_rate = mapped_count / total_rows * 100 if total_rows > 0 else 0
    
    print(f"\nProcessing Complete!")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Rows with mappings: {mapped_count:,} ({mapping_rate:.1f}%)")
    print(f"  Processing time: {total_time/60:.1f} minutes")
    print(f"  Output file: {output_file}")