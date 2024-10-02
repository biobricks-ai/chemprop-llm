import numpy as np
import json
import re
import dotenv
import os
import pyspark
import biobricks as bb
import pandas as pd
import pathlib
from openai import OpenAI
from functools import lru_cache
from joblib import Memory
import shutil
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set your OpenAI API key
dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def o1_preview(prompt):
    completion = client.chat.completions.create(
        model="o1-preview-2024-09-12",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


def lc50_query(prompt, units):
    json_schema = {
        "name": "lc50_4hr_" + units,
        "schema": {
            "type": "object",
            "properties": {
                "lc50_4hr_" + units: {
                    "type": "number",
                    "description": "The 4-hour LC50 value in " + units
                },
                'confidence': {
                    'type': 'number',
                    'description': 'The confidence of the estimate between 1 and 10, test the most confident'
                }
            },
            "required": ["lc50_4hr_" + units, "confidence"],
            "additionalProperties": False
        },
        "strict": True
    }

    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
        response_format={
        "type": "json_schema",
        "json_schema": json_schema
            }
        )
    try:
        safe_json = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        safe_json = {"lc50_4hr_" + units: None, "confidence": 0}
    
    return safe_json

