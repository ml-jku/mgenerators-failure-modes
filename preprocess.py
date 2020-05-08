import numpy as np
import pandas as pd


def prepjak2(write=False):
    chid = 'CHEMBL3888429'
    df = pd.read_csv(f'./assays/raw/{chid}.csv', sep=';')
    df = df[['Smiles', 'pChEMBL Value']]
    df.columns = ['smiles', 'value']

    label = np.array([1 if x > 8 else 0 for x in df.value])
    df['label'] = label

    if write:
        df.to_csv(f'./assays/processed/{chid}.csv', index=None)
    return df


def prepegfr(write=False):
    chid = 'CHEMBL1909203'
    df = pd.read_csv(f'./assays/raw/{chid}.csv', sep=';')
    df.head()

    # df[]
    df['label'] = pd.isna(df['Comment']).astype('int')
    df['smiles'] = df['Smiles']
    df = df[['smiles', 'label']]
    df = df.dropna()
    if write:
        df.to_csv(f'./assays/processed/{chid}.csv', index=None)
    return df


def prepdrd2(write=False):
    chid = 'CHEMBL1909140'
    df = pd.read_csv(f'./assays/raw/{chid}.csv', sep=';')
    df.head()

    # df[]
    df['label'] = pd.isna(df['Comment']).astype('int')
    df['smiles'] = df['Smiles']
    df = df[['smiles', 'label']]
    df = df.dropna()
    df.to_csv(f'./assays/processed/{chid}.csv', index=None)
    return df


if __name__ == '__main__':
    prepegfr(write=True)
    prepjak2(write=True)
    prepdrd2(write=True)
