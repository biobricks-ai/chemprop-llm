import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm  

output_folder = Path("path/to/processed/output")
processed_tables_folder = output_folder.parent / "processed_tables_nodescribe"
processed_tables_folder.mkdir(parents=True, exist_ok=True)

def serialize_value(value):
    return str(value)

# Function to extract property-value pairs from the Data column
def extract_properties(data, source_name, description, heading):
    try:
        data_json = json.loads(data)  # Parse the JSON data
        property_value_pairs = []
        base_property = f"{source_name} - {heading}"
        value_section = data_json.get("Value", {})
        
        # Case 1: Handle if "StringWithMarkup" exists in Value
        if "StringWithMarkup" in value_section:
            for markup in value_section["StringWithMarkup"]:
                value = markup.get("String", None)
                if value:
                    property_value_pairs.append({
                        'property': base_property,
                        'value': serialize_value(value)
                    })
        
        # Case 2: Extract other key-value pairs in the "Value" section
        for key, val in value_section.items():
            if key != "StringWithMarkup":  # Avoid duplicate entries
                property_value_pairs.append({
                    'property': f"{base_property} - {key}",
                    'value': serialize_value(val)
                })

        return property_value_pairs
    
    except Exception as e:
        print(f"Error parsing data: {e}")
        return []

processed_files = list(output_folder.glob('processed_*.parquet'))

if processed_files:
    for parquet_file in tqdm(processed_files, desc="Processing files", unit="file"):
        sample_df = pd.read_parquet(parquet_file)
        property_value_rows = []
        for _, row in sample_df.iterrows():
            # Extract properties by combining SourceName, (Description), heading, and Data
            properties = extract_properties(row['Data'], row['SourceName'], row['Description'], row['heading'])
            for prop in properties:
                property_value_rows.append({
                    'selfies': row['selfies'],
                    'selfies_split': row['selfies_split'],  # Use the existing 'selfies_split' column
                    'property': prop['property'],
                    'value': serialize_value(prop['value'])  # Serialize to string
                })

        exploded_df = pd.DataFrame(property_value_rows)
        output_file = processed_tables_folder / f"{parquet_file.stem}_exploded.parquet"
        exploded_df.to_parquet(output_file, compression='snappy')
        tqdm.write(f"Processed and saved: {output_file}")

else:
    print(f"No processed files found in the output folder: {output_folder}")
# ===========Show examples of the processed data from the exploded tables============
processed_tables = list(processed_tables_folder.glob('*_exploded.parquet'))
if processed_tables:
    sample_file = processed_tables[0]
    print(f"\nShowing example from processed table: {sample_file.name}")
    df = pd.read_parquet(sample_file)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
    print("\nDataFrame Columns:")
    print(df.columns.tolist())
    print("\nBasic Statistics:")
    print(df.describe())
else:
    print("No processed exploded tables available to show examples.")

