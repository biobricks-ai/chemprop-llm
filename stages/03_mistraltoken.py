import os
import json
import pandas as pd
import pathlib
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
import biobricks as bb

# Load chemprop assets from Biobricks to get the path to the SELFIES tokenizer
chemprop = bb.assets("chemprop-transformer")
selfies_tokenizer_file = pathlib.Path(chemprop.cvae_sqlite).parent / 'selfies_property_val_tokenizer' / 'selfies_tokenizer.json'

# Load SELFIES symbols from the JSON file
if selfies_tokenizer_file.exists():
    with open(selfies_tokenizer_file, 'r') as f:
        selfies_tokenizer_data = json.load(f)
    selfies_symbols = list(selfies_tokenizer_data.get('symbol_to_index', {}).keys())
else:
    raise FileNotFoundError(f"selfies_tokenizer.json file not found: {selfies_tokenizer_file}")

selfies_symbols = [symbol for symbol in selfies_symbols if symbol not in ['<pad>', '<sos>', '<eos>']]
# Set environment variable to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "mistralai/Mistral-7B-v0.1"  # Mistral-7B model
tokenizer = AutoTokenizer.from_pretrained(model_id)
num_added = tokenizer.add_tokens(selfies_symbols)
print(f"Number of SELFIES tokens added to the tokenizer: {num_added}")

# Verify that the SELFIES symbols were added to the tokenizer
added_tokens = tokenizer.convert_tokens_to_ids(selfies_symbols)
successful_additions = [symbol for symbol, token_id in zip(selfies_symbols, added_tokens) if token_id != tokenizer.unk_token_id]
failed_additions = [symbol for symbol, token_id in zip(selfies_symbols, added_tokens) if token_id == tokenizer.unk_token_id]
print(f"Successfully added {len(successful_additions)} SELFIES symbols")
print(f"Failed to add {len(failed_additions)} SELFIES symbols")
print(f"New vocabulary size: {len(tokenizer)}")

# Define input and output folders
input_folder = Path("path/to/processed/processed_tables_nodescribe")
output_folder = input_folder.parent / "tokenized_table"
output_folder.mkdir(parents=True, exist_ok=True)
parquet_files = list(input_folder.glob("*.parquet"))

for parquet_file in tqdm(parquet_files, desc="Processing files"):
    df = pd.read_parquet(parquet_file)
    required_columns = ['selfies_split', 'property', 'value']
    
    if all(col in df.columns for col in required_columns):
        
        df['split_selfies_tokenized'] = [
            tokenizer.encode(str(x), max_length=None)  # No truncation during tokenization
            for x in tqdm(df['selfies_split'], desc=f"Tokenizing split_selfies from {parquet_file.stem}")
        ]

        df['property_tokenized'] = [
            tokenizer.encode(str(x), max_length=2048)[:2048]  # Truncate to 2048 tokens after tokenization
            for x in tqdm(df['property'], desc=f"Tokenizing property from {parquet_file.stem}")
        ]

        df['value_tokenized'] = [
            tokenizer.encode(str(x), max_length=2048)[:2048]  # Truncate to 2048 tokens after tokenization
            for x in tqdm(df['value'], desc=f"Tokenizing value from {parquet_file.stem}")
        ]

        output_file = output_folder / f"{parquet_file.stem}_tokenized.parquet"
        df.to_parquet(output_file, compression='snappy')
        print(f"Tokenized file saved to: {output_file}")
    else:
        missing_columns = [col for col in required_columns if col not in df.columns]
        print(f"Skipping file {parquet_file}: Missing columns {', '.join(missing_columns)}")

# Optional: Summary of the process
print(f"\nProcessed {len(parquet_files)} files. Tokenized files saved in {output_folder}.")

# =========Read one file from the tokenized table folder as an example and display its contents=================
tokenized_files = list(output_folder.glob("*_tokenized.parquet"))

if tokenized_files:
    example_file = tokenized_files[0]
    print(f"\nReading example tokenized file: {example_file.name}")
    example_df = pd.read_parquet(example_file)
    print("\nDataFrame Info:")
    example_df.info()
    print("\nFirst few rows:")
    print(example_df.head())
    print("\nColumn names:")
    print(example_df.columns.tolist())
    print("\nTokenized column statistics:")
    for col in ['split_selfies_tokenized', 'property_tokenized', 'value_tokenized']:
        if col in example_df.columns:
            token_lengths = example_df[col].apply(len)
            print(f"\n{col}:")
            print(f"  Min length: {token_lengths.min()}")
            print(f"  Max length: {token_lengths.max()}")
            print(f"  Mean length: {token_lengths.mean():.2f}")
            print(f"  Median length: {token_lengths.median()}")
else:
    print("No tokenized files found in the output folder.")
