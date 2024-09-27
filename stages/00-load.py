# purpose: update a llm tokenizer with selfies tokens
from transformers import AutoTokenizer
import biobricks as bb
import os, sys, pathlib
import pyspark.sql, pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, IntegerType
import stages.utils.spark_helpers as H, stages.utils.spark_selfies_tokenizer as st
import importlib
importlib.reload(H)
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

spark = pyspark.sql.SparkSession.builder \
    .appName("ChemharmonyDataProcessing") \
    .config("spark.driver.memory", "64g") \
    .config("spark.driver.maxResultSize", "48g") \
    .config("spark.executor.memory", "64g") \
    .getOrCreate()

# Load compait data
compait = bb.assets('compait')
compounds_df = spark.read.parquet(compait.PredictionSet_parquet)
LC50_df = spark.read.parquet(compait.LC50_Tr_parquet)

# Add a new column 'selfies' to the LC50 DataFrame by converting 'QSAR_READY_SMILES' to selfies
LC50_df = LC50_df.withColumn('selfies', H.smiles_to_selfies_udf('QSAR_READY_SMILES'))

# Create DataFrames for each property with better property names
mgl_property = 'Lethal concentration where 50% of the population dies during a 4-hour exposure of inhalation, AKA 4hr_LC50 (mg/L)'
mgl_df = LC50_df.select('selfies', F.lit(mgl_property).alias('property'), '4_hr_value_mgL') \
    .filter(F.col('4_hr_value_mgL').isNotNull()) \
    .withColumnRenamed('4_hr_value_mgL', 'value')

ppm_property = 'Parts of the chemical per million where 50% of the population dies during a 4-hour exposure of inhalation, AKA 4hr_LC50 (ppm)'
ppm_df = LC50_df.select('selfies', F.lit(ppm_property).alias('property'), '4_hr_value_ppm') \
    .filter(F.col('4_hr_value_ppm').isNotNull()) \
    .withColumnRenamed('4_hr_value_ppm', 'value')

# Concatenate the two DataFrames and split the selfies
combined_df = mgl_df.union(ppm_df).withColumn('selfies_split', H.split_selfies_udf('selfies'))

# Load pubchem-annotations data
pubchem_annotations = bb.assets('pubchem-annotations')
annotations_df = spark.read.parquet(pubchem_annotations.annotations_parquet).cache()

annotations_df = annotations_df.filter(F.size('PubChemCID') == 1)
annotations_df = annotations_df.withColumn('PubChemCID', F.col('PubChemCID')[0])
unique_cids = annotations_df.select('PubChemCID').distinct()

pubchem_biobrick = bb.assets('pubchem')
pubchem_df= spark.read.parquet(pubchem_biobrick.compound_sdf_parquet)

# select pubchem_df to only include rows where the column id is in the unique_cids DataFrame
pubchem_df = pubchem_df.join(unique_cids, pubchem_df.id == unique_cids.PubChemCID, 'inner')

# Keep only rows in the 'property' column that contain the word 'SMILES'
filtered_pubchem_df = pubchem_df.filter(F.col('property').contains('SMILES'))

# Convert the 'value' column from SMILES to SELFIES using the UDF
filtered_pubchem_df = filtered_pubchem_df.withColumn('selfies', H.smiles_to_selfies_udf('value'))

# Use the split_selfies_udf to split the 'selfies' column into individual symbols
filtered_pubchem_df = filtered_pubchem_df.withColumn('selfies_split', H.split_selfies_udf('selfies'))

# Combine only selfies_split columns from both DataFrames
combined_selfies_split = combined_df.select('selfies_split').union(filtered_pubchem_df.select('selfies_split'))

# Explode the combined_selfies_split DataFrame to get individual symbols
exploded_selfies_split = combined_selfies_split.select(F.explode('selfies_split').alias('symbol'))

# Get a collection of all unique elements
unique_elements = exploded_selfies_split.distinct()


# TODO 