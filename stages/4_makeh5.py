import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Load tokenized Parquet files
tokenized_folder = Path("path/to/processed/tokenized_table") 
hdf5_file = "chem_data_tokenized.h5"
tokenized_files = list(tokenized_folder.glob("*_tokenized.parquet"))

# # Limit the number of tokenized files to a small subset for testing (e.g., first 3 files)
# test_tokenized_files = tokenized_files[:3]  # Change the number for more files

def store_tokenized_data_in_hdf5(tokenized_files, hdf5_file):
    print(f"Number of tokenized files found: {len(tokenized_files)}")

    with h5py.File(hdf5_file, 'w') as hdf:
        selfies_ds = hdf.create_dataset('selfies_split_tokenized', (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.int32))
        properties_ds = hdf.create_dataset('property_tokenized', (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.int32))
        values_ds = hdf.create_dataset('value_tokenized', (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.int32))

        for tokenized_file in tqdm(tokenized_files, desc="Storing tokenized data (files)"):
            print(f"Processing file: {tokenized_file}")
            df = pd.read_parquet(tokenized_file)
            print(df.head())  # Print a few rows to check the contents

            selfies_data = df['split_selfies_tokenized'].tolist()
            property_data = df['property_tokenized'].tolist()
            value_data = df['value_tokenized'].tolist()

            # Show progress for data processing
            for idx in tqdm(range(len(selfies_data)), desc="Processing tokenized data (rows)", leave=False):
                selfies_ds.resize((selfies_ds.shape[0] + 1,))
                properties_ds.resize((properties_ds.shape[0] + 1,))
                values_ds.resize((values_ds.shape[0] + 1,))

                selfies_ds[-1] = selfies_data[idx]
                properties_ds[-1] = property_data[idx]
                values_ds[-1] = value_data[idx]

        # Convert tokenized selfies data to tuples and sort to create an index
        selfies_as_tuples = [tuple(seq) for seq in selfies_ds]

        # Sort the tuples and store the indices
        sorted_index = sorted(range(len(selfies_as_tuples)), key=lambda i: selfies_as_tuples[i])

        # Store the sorted index as a dataset instead of an attribute
        hdf.create_dataset('selfies_index', data=sorted_index)

# Run the function with the test subset
store_tokenized_data_in_hdf5(tokenized_files, hdf5_file)

def load_selfies_data_from_hdf5(file_name, query_selfies_tokenized):
    with h5py.File(file_name, 'r') as hdf:
        selfies_ds = hdf['selfies_split_tokenized']
        properties_ds = hdf['property_tokenized']
        values_ds = hdf['value_tokenized']
        selfies_index = hdf['selfies_index'][:]  # Load the sorted index dataset
        
        # Convert the tokenized query selfies to a tuple for comparison
        query_tuple = tuple(query_selfies_tokenized)
        
        # Fetch sorted selfies sequences in tuple form
        sorted_selfies = [tuple(selfies_ds[i]) for i in selfies_index]
        
        # Perform manual search for query_tuple in sorted_selfies
        for idx, selfies_tuple in enumerate(sorted_selfies):
            if selfies_tuple == query_tuple:
                original_idx = selfies_index[idx]
                prop = properties_ds[original_idx]
                value = values_ds[original_idx]
                return prop, value
        
        # If not found, return None
        return None, None

# Example query
# remeber to load tokenizer
tokenized_query = tokenizer.encode('[C][C][O][H]', max_length=None)
prop, value = load_selfies_data_from_hdf5(hdf5_file, tokenized_query)
print(f"Property: {prop}, Value: {value}")
