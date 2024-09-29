import json

def process_stringwithmarkup(data):
    stringwithmarkup = data['Value']['StringWithMarkup']
    processed = [item['String'] for item in stringwithmarkup if 'String' in item]
    processed_string = ' '.join(processed)
    return processed_string

def process_data(data_string):
    data = json.loads(data_string)
    if 'Value' in data and 'Number' in data['Value']:
        return data['Value']['Number'][0]
    elif 'Value' in data and 'StringWithMarkup' in data['Value']:
        return process_stringwithmarkup(data)
    else:
        return None