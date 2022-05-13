from google.cloud import storage
import pandas as pd
from tqdm.auto import tqdm


def prepare_30s_resample(filename='sample30sec.csv', BUCKET_NAME = 'gcp101227-sfpdemo-datasets',
                         skip_if_exist: bool = True, save_to_disk: bool = False):
    path = 'gs://' + BUCKET_NAME + '/' + filename
    try:
        df = pd.read_csv(path, parse_dates=['SensorDateTime'])
        print("Read from cache:", path)
        if save_to_disk:
            df.to_csv(f'data/{filename}', index=False)
        return df
    except FileNotFoundError:
        print("Failed to read preprocessed data from cache:", path)
        pass

    client = storage.Client()

    bucket = client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()
    csv_files = [file.name for file in blobs if '.csv' in file.name and 'hipe/clean' in file.name]
    file_name = csv_files[0]
    path_file = 'gs://' + BUCKET_NAME + '/' + file_name
    
    
    df = []
    for i in tqdm(csv_files):
        temp = pd.read_csv('gs://' + BUCKET_NAME + '/' +i)
        temp['SensorDateTime'] = pd.to_datetime(temp['SensorDateTime'], utc=True)
        summary = temp.resample('30S', on='SensorDateTime').mean()
        summary['SensorDateTime'] = summary.index
        summary['Machine'] = temp['Machine'][0]
        summary.reset_index(inplace=True, drop=True)
        df.append(summary)

    df = pd.concat(df)
    df.reset_index(inplace=True, drop=True)

    wide = df.pivot(index='SensorDateTime', columns='Machine', values='P_kW')

    wide.to_csv('gs://' + BUCKET_NAME + '/'+filename, index=True)
    if save_to_disk:
        wide.to_csv(f'data/{filename}', index=False)
    return wide


def prepare_1m_resample(filename='sample1min.csv', BUCKET_NAME='gcp101227-sfpdemo-datasets', save_to_disk: bool = False):
    path = 'gs://' + BUCKET_NAME + '/' + filename
    try:
        df = pd.read_csv(path, parse_dates=['SensorDateTime'])
        print("Read from cache:", path)
        if save_to_disk:
            df.to_csv(f'data/{filename}', index=False)
        return df
    except FileNotFoundError:
        print("Failed to read preprocessed data from cache:", path)
        pass

    df = []
    for i in csv_files:
        temp = pd.read_csv('gs://' + BUCKET_NAME + '/' +i)
        temp['SensorDateTime'] = pd.to_datetime(temp['SensorDateTime'], utc=True)
        summary = temp.resample('1Min', on='SensorDateTime').max()
        summary['SensorDateTime'] = summary.index
        summary['Machine'] = temp['Machine'][0]
        summary.reset_index(inplace=True, drop=True)
        df.append(summary)

    df = pd.concat(df)
    df.reset_index(inplace=True, drop=True)
    
    wide = df.pivot(index='SensorDateTime', columns='Machine', values='P_kW')

    wide.to_csv('gs://' + BUCKET_NAME + '/'+filename, index=True)
    if save_to_disk:
        wide.to_csv(f'data/{filename}', index=False)
    return wide
