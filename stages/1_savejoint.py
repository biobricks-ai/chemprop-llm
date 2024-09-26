import biobricks as bb
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import selfies
from selfies.exceptions import SMILESParserError, EncoderError
from rdkit import Chem
from rdkit import RDLogger
# maybe I can use udf but for now keep as is
def smiles_to_selfies(smiles):
    RDLogger.DisableLog('rdApp.*')
    try:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return selfies.encoder(smiles)
    except (SMILESParserError, EncoderError):
        return None
    return None

def split_selfies(selfies_string):
    return list(selfies.split_selfies(selfies_string)) if selfies_string else []

pubchem_annotations = bb.assets('pubchem-annotations')
annotations_df = pd.read_parquet(pubchem_annotations.annotations_parquet)

# Filter annotations to ensure `PubChemCID` contains exactly one CID per row
# maybe we look into why annotations_df has multiple CIDs per row
annotations_df = annotations_df[annotations_df['PubChemCID'].apply(len) == 1]
annotations_df['PubChemCID'] = annotations_df['PubChemCID'].apply(lambda x: x[0])

# Set up the directory to save the processed files
# output_folder = Path("path/to/processed/output")
# output_folder.mkdir(parents=True, exist_ok=True)

pubchem_biobrick = bb.assets('pubchem')
parquet_files = [f for f in Path(pubchem_biobrick.compound_sdf_parquet).glob('*.parquet')]

# Process each Parquet file using Pandas
parquet_files_with_progress = tqdm(parquet_files, desc="Processing files", unit="file")
for parquet_file in parquet_files_with_progress:
    parquet_files_with_progress.set_postfix({"File": parquet_file.name})
    df = pd.read_parquet(parquet_file)
    df = df.merge(annotations_df, left_on='id', right_on='PubChemCID', how='inner')
    df = df[df['property'] == 'PUBCHEM_OPENEYE_CAN_SMILES']
    df['selfies'] = df['value'].apply(smiles_to_selfies)
    df['selfies_split'] = df['selfies'].apply(split_selfies)
    df = df.drop(columns=['id', 'property', 'value'], errors='ignore')
    output_file = output_folder / f"processed_{parquet_file.name}"
    df.to_parquet(output_file, compression="snappy")

print("Processing complete.")

# ===Optional: Display sample data from processed files ===
processed_files = list(output_folder.glob('processed_*.parquet'))

if processed_files:
    sample_file = processed_files[30]
    sample_df = pd.read_parquet(sample_file)
    print(f"Sample data from {sample_file.name}:")
    print(sample_df.head())
    print("\nDataFrame info:")
    sample_df.info()
else:
    print(f"No processed files found in the output folder: {output_folder}")
# >>> sample_df.columns
# Index(['SourceName', 'SourceID', 'Name', 'Description', 'URL', 'LicenseNote',
#        'LicenseURL', 'heading', 'type', 'DataName', 'Data', 'PubChemCID',
#        'PubChemSID', 'selfies', 'selfies_split'],
#       dtype='object')

