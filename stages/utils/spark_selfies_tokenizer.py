import json
import selfies as sf
from pyspark.sql.types import BooleanType, ArrayType, IntegerType, FloatType, StringType
import pyspark.sql.functions as F

class SelfiesTokenizer:
    
    def __init__(self, unknown_token='<unk>', unknown_token_id=0):
        self.unknown_token = unknown_token
        self.unknown_token_id = unknown_token_id
        self.symbol_to_index = {unknown_token: unknown_token_id}
        self.index_to_symbol = {unknown_token_id: unknown_token}

    def fit(self, dataset, column):
        # Extract unique symbols from the dataset in a distributed manner
        unique_symbols_rdd = dataset.select(column).rdd \
            .flatMap(lambda row: sf.split_selfies(row[column]) if row[column] is not None else []) \
            .distinct()

        # Collect unique symbols to the driver and merge with existing mappings
        unique_symbols = unique_symbols_rdd.collect()
        start_idx = len(self.symbol_to_index)  # Start indexing after special tokens
        new_mappings = {symbol: idx + start_idx for idx, symbol in enumerate(unique_symbols) if symbol not in self.symbol_to_index}
        self.symbol_to_index.update(new_mappings)
        self.index_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_index.items()}

        return self

    def selfies_to_indices(self, selfies_string):
        symbols = list(sf.split_selfies(selfies_string))
        indices = [self.symbol_to_index.get(symbol, self.symbol_to_index[self.pad_token]) for symbol in symbols]
        return indices
            
    def transform(self, dataset, selfies_column, new_column, pad_length=120):
        def selfies_to_indices(selfies_string):
            if selfies_string is not None:
                symbols = list(sf.split_selfies(selfies_string))
                indices = [self.symbol_to_index.get(symbol, self.symbol_to_index[self.pad_token]) for symbol in symbols]
                padded_indices = indices[:pad_length] + [self.symbol_to_index[self.pad_token]] * max(0, pad_length - len(indices))
                return padded_indices[:pad_length]
            else:
                return [self.symbol_to_index[self.pad_token]] * pad_length

        selfies_to_indices_udf = F.udf(selfies_to_indices, ArrayType(IntegerType()))

        return dataset.withColumn(new_column, selfies_to_indices_udf(F.col(selfies_column)))