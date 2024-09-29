# purpose: load pubchem data and save training table selfies, property, value also output some helper files
# output: cache/01_load_pubchem/pubchem_train.parquet, cache/01_load_pubchem/pubchem_id_mapping.parquet
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

import biobricks as bb, pandas as pd, sys, json, pyspark, pyspark.sql, pyspark.sql.functions as F, shutil
sys.path.append('./')
import stages.utils.spark_helpers as sh
import stages.utils.pubchem_annotations as pubchem_annotations

outdir = Path("cache/01_load_pubchem")
outdir.mkdir(parents=True, exist_ok=True)

propvalpath = outdir / "selfies_property_value.parquet"
shutil.rmtree(propvalpath, ignore_errors=True)
propvalpath.mkdir(parents=True, exist_ok=True)

compait_tst_path = outdir / "compait_tst.parquet"
shutil.rmtree(compait_tst_path, ignore_errors=True)
compait_tst_path.mkdir(parents=True, exist_ok=True)

spark = pyspark.sql.SparkSession.builder \
    .appName("ChemharmonyDataProcessing") \
    .config("spark.driver.memory", "64g") \
    .config("spark.driver.maxResultSize", "48g") \
    .config("spark.executor.memory", "64g") \
    .getOrCreate()

# load pubchem annotations which are needed for compait and pubchem_annotations
pc_annotations = bb.assets('pubchem-annotations')
idannotations = pd.read_parquet(pc_annotations.annotations_parquet)
idannotations = idannotations[idannotations['PubChemCID'].progress_apply(len) == 1]
idannotations['PubChemCID'] = idannotations['PubChemCID'].progress_apply(lambda x: x[0])
idannotations['value'] = idannotations['Data'].progress_apply(pubchem_annotations.process_data)

pcid = idannotations[idannotations['SourceName'] == 'EPA DSSTox']
pcid = pcid[['PubChemCID','value']].drop_duplicates().reset_index(drop=True)
pcid.columns = ['PubChemCID', 'dsstox']

# LOAD COMPAIT =========================================================================================================
pcid_spark = spark.createDataFrame(pcid).withColumnRenamed('dsstox', 'DSSTOX_SUBSTANCE_ID')

compait = bb.assets('compait')

compait_tst = spark.read.parquet(compait.PredictionSet_parquet).select('DSSTOX_SUBSTANCE_ID')
compait_tst = compait_tst.join(pcid_spark, on='DSSTOX_SUBSTANCE_ID', how='left')
compait_tst = compait_tst.toPandas()

compait_trn = spark.read.parquet(compait.LC50_Tr_parquet).select('DTXSID',"4_hr_value_mgl","4_hr_value_ppm")
compait_trn = compait_trn.withColumnRenamed('DTXSID', 'DSSTOX_SUBSTANCE_ID')
compait_trn = compait_trn.join(pcid_spark, on='DSSTOX_SUBSTANCE_ID', how='left')

# Create DataFrames for each property with better property names
mgl_property = 'Lethal concentration where 50% of the population dies during a 4-hour exposure of inhalation, AKA 4hr_LC50 (mg/L)'
mgl_df = compait_trn \
    .filter(F.col('4_hr_value_mgL').isNotNull()) \
    .withColumnRenamed('4_hr_value_mgL', 'value') \
    .withColumn('property', F.lit(mgl_property)) \
    .select('PubChemCID','DSSTOX_SUBSTANCE_ID', 'property', 'value')

ppm_property = 'Parts of the chemical per million where 50% of the population dies during a 4-hour exposure of inhalation, AKA 4hr_LC50 (ppm)'
ppm_df = compait_trn \
    .filter(F.col('4_hr_value_ppm').isNotNull()) \
    .withColumnRenamed('4_hr_value_ppm', 'value') \
    .withColumn('property', F.lit(ppm_property)) \
    .select('PubChemCID','DSSTOX_SUBSTANCE_ID', 'property', 'value')

# Concatenate the two DataFrames and split the selfies
compait_train_total = mgl_df.union(ppm_df).toPandas()
compait_train_total = compait_train_total[compait_train_total['PubChemCID'].notna()]
compait_train_total['PubChemCID'] = compait_train_total['PubChemCID'].astype('int64')

# MERGE ANNOTATIONS AND SMILES AND CREATE SELFIES ======================================================================
# output: cache/01_load_pubchem/pubchem_id_mapping.parquet

helpful_headers = [line.strip().lower() for line in open('stages/resources/helpful_headers.txt', 'r')]
has_helpful_header = [
    any(header in heading.lower() for header in helpful_headers)
    for heading in tqdm(idannotations['heading'], desc="Checking headers", unit="header")
]
pcannotations = idannotations[has_helpful_header]

create_property = lambda row: f"{row['SourceName']} - {row['heading']} - {row['name']}"
pcannotations['name'] = pcannotations['Data'].progress_apply(lambda x: json.loads(x).get('Name', ''))
pcannotations['property'] = pcannotations.progress_apply(create_property, axis=1)
pcannotations = pcannotations[['PubChemCID','property','value']]

pubchem_biobrick = bb.assets('pubchem')
parquet_files = [f for f in Path(pubchem_biobrick.compound_sdf_parquet).glob('*.parquet')]
for parquet_file in tqdm(parquet_files, desc="Processing files", unit="file"):
    df = pd.read_parquet(parquet_file).rename(columns={'id': 'PubChemCID'})
    df = df[df['property'] == 'PUBCHEM_OPENEYE_CAN_SMILES'].rename(columns={'value': 'smiles'})
    df = df.drop(columns=['property'])
    df['PubChemCID'] = df['PubChemCID'].astype('int64')
    
    # create pc_annotations
    pcannodf = df.merge(pcannotations, left_on='PubChemCID', right_on='PubChemCID', how='inner')
    compaitdf = df.merge(compait_train_total, left_on='PubChemCID', right_on='PubChemCID', how='inner')
    union_df = pd.concat([pcannodf, compaitdf])
    union_df['selfies'] = union_df['smiles'].progress_apply(sh.smiles_to_selfies_safe)
    union_df['selfies_split'] = union_df['selfies'].progress_apply(sh.split_selfies)
    union_df = union_df[['PubChemCID','DSSTOX_SUBSTANCE_ID','selfies','selfies_split','property','value']]
    
    # Convert 'value' column to numeric type
    union_df['value'] = union_df['value'].astype(str)    
    union_df.to_parquet(propvalpath / parquet_file.name, compression="snappy")

    compait_compounds = df.merge(compait_tst, left_on='PubChemCID', right_on='PubChemCID', how='inner')
    compait_compounds['selfies'] = compait_compounds['smiles'].progress_apply(sh.smiles_to_selfies_safe)
    compait_compounds['selfies_split'] = compait_compounds['selfies'].progress_apply(sh.split_selfies)
    compait_compounds = compait_compounds[['PubChemCID','DSSTOX_SUBSTANCE_ID','selfies','selfies_split']]
    compait_compounds.to_parquet(compait_tst_path / parquet_file.name, compression="snappy")

# merge all the propvalpath files
propvaldf = pd.concat([pd.read_parquet(f) for f in tqdm(list(propvalpath.glob('*.parquet')), desc="Merging files", unit="file")])
shutil.rmtree(propvalpath)
propvaldf.to_parquet(propvalpath, compression="snappy")

# merge all the compait_tst files
outcompaitdf = pd.concat([pd.read_parquet(f) for f in tqdm(list(compait_tst_path.glob('*.parquet')), desc="Merging files", unit="file")])
shutil.rmtree(compait_tst_path)
outcompaitdf.to_parquet(compait_tst_path, compression="snappy")
