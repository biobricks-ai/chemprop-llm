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


def lc50_query_new(units, prompt, chemical_name):
    json_schema = {
        "name": "lc50",
        "schema": {
            "type": "object",
            "properties": {
                "chemical_name": {
                    "type": "string",
                    "description": f"The chemical name of the compound"
                },
                "units": {
                    "type": "string",
                    "description": f"The units of the LC50 value, which should be {units}"
                },
                "explanation": {
                    "type": "string",
                    "description": f"A short explanation of how the LC50 value was estimated"
                },
                "lc50": {
                    "type": "number",
                    "description": f"The 4 hour LC50 value in rats."
                },
                'confidence': {
                    'type': 'number',
                    'description': 'The confidence of the estimate between 1 and 10, 10 being most confident'
                }
            },
            "required": ["chemical_name", "units", "explanation", "lc50", "confidence"],
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

def lc50_query_ppm(units, prompt, chemical_name):
    json_schema = {
        "name": "lc50",
        "schema": {
            "type": "object",
            "properties": {
                "chemical_name": {
                    "type": "string",
                    "description": "The chemical name of the compound"
                },
                "units": {
                    "type": "string",
                    "description": f"The units of the LC50 value, which should be {units}"
                },
                "explanation": {
                    "type": "string",
                    "description": f"A short explanation of how the LC50 value was estimated"
                },
                "lc50": {
                    "type": "number",
                    "description": f"The 4-hour LC50 value in parts per million(ppm) for rats."
                },
                'confidence': {
                    'type': 'number',
                    'description': 'The confidence of the estimate between 1 and 10, 10 being most confident'
                }
            },
            "required": ["chemical_name", "units", "explanation", "lc50", "confidence"],
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



def refine_prompt(prompt):
    """
    Refines the initial prompt to improve its suitability for acute inhalation LC50 estimation.
    
    Parameters:
        prompt (str): The initial prompt text including chemical details.
        
    Returns:
        str: The refined prompt for the most accurate acute inhalation LC50 estimation.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}]
        )
        refined_prompt = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error refining prompt: {e}")
        refined_prompt = prompt  # Fallback to the original prompt if an error occurs
    
    return refined_prompt

