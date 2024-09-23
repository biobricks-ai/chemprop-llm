from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, ArrayType, IntegerType, FloatType, StringType
import pyspark.sql.functions as F
import selfies
from selfies.exceptions import EncoderError, SMILESParserError
from rdkit import Chem
from rdkit import RDLogger

def is_valid_smiles(smiles):
    """Validate SMILES string using RDKit."""
    RDLogger.DisableLog('rdApp.*') # Disable RDKit warnings
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and mol.GetNumAtoms() > 0
    except:
        return False

# Register the UDF
is_valid_smiles_udf = udf(is_valid_smiles, BooleanType())

# INCHI TO SMILES UDF ============================================================================
def inchi_to_smiles_safe(inchi):
    RDLogger.DisableLog('rdApp.*') # Disable RDKit warnings
    try:
        if inchi:
            mol = Chem.MolFromInchi(inchi)
            if mol:
                return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error in converting InChI to SMILES: {e}")
    return None

inchi_to_smiles_udf = udf(inchi_to_smiles_safe, StringType())

# SMILES TO SELFIES UDF ===========================================================================
def smiles_to_selfies_safe(smiles):
    RDLogger.DisableLog('rdApp.*') # Disable RDKit warnings
    try:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return selfies.encoder(smiles)
    except SMILESParserError as e:
        return None
    except EncoderError as e:
        return None
    return None

smiles_to_selfies_udf = udf(smiles_to_selfies_safe, StringType())

# SELFIES SYMBOL SPLITTER UDF ====================================================================
def split_selfies(selfies_string):
    return list(selfies.split_selfies(selfies_string)) if selfies_string else []

split_selfies_udf = udf(split_selfies, ArrayType(StringType()))


