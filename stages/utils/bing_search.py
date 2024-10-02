import os, tqdm
import requests
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import sys
sys.path.append('./')
import stages.utils.scraperapi as scraperapi


# Load environment variables
load_dotenv()
bing_api_key = os.getenv('BINGAPI_KEY')
query = "acute inhalation toxicity \"LC50\" PPM mg \"sesquimustard\""
def bing_search(query, count, offset=0):
    headers = {'Ocp-Apim-Subscription-Key': bing_api_key}
    params = {'q': query, 'mkt': 'en-US', 'count': count, 'offset': offset }
    response = requests.get('https://api.bing.microsoft.com/v7.0/search', headers=headers, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors
    results = response.json()
    urls = []
    for item in results.get('webPages', {}).get('value', []):
        license_info = next((rule['license'] for rule in item.get('contractualRules', []) if rule['_type'] == 'ContractualRules/LicenseAttribution'), None)
        urls.append({
            'query' : query,
            'name': item.get('name'),
            'url': item.get('url'),
            'snippet': item.get('snippet'),
            'license_name': license_info['name'] if license_info else '',
            'license_url': license_info['url'] if license_info else ''
        })
    return urls