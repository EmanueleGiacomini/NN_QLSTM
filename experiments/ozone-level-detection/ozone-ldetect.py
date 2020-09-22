""""ozone-ldetect.py

The script uses the following dataset:
https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection

Two ground ozone level data sets are included in this collection.
One is the eight hour peak set (eighthr.data), the other is the one hour peak set (onehr.data).
Those data were collected from 1998 to 2004 at the Houston, Galveston and Brazoria area.
"""

import pandas as pd

def extractNames(path: str) -> [str]:
    labels = []
    with open(path, mode='r') as f:
        # Read lines to obtain labels from the file
        # Skip the first two rows
        f.readline()
        f.readline()
        for line in f.readlines():
            label = line.strip().split(':')[0]
            labels.append(label)
    # Append the class label
    labels.append('CLASS')
    return labels

def getData(path: str, col_names_path: str) -> pd.DataFrame:
    col_names = extractNames(col_names_path)
    df = pd.read_table('onehr.data', delimiter=',', names=col_names)
    return df

def generateSequence(df: pd.DataFrame, seq_len: int):
    """Generates a set of sequences. Each element of the set
     is formed by a sequence of readings"""
    pass
NAMES_PATH = 'eighthr.names'
DATA_PATH = 'onehr.data'

if __name__ == '__main__':
    df = getData(DATA_PATH, NAMES_PATH)

    exit(0)