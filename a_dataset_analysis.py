import datetime
import os
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def calc_variance_threshold(path):
    dataset_sensors = pd.read_csv(path, engine="pyarrow")
    dataset_sensors = dataset_sensors.dropna()
    dataset_sensors = dataset_sensors[0:int(len(dataset_sensors) * 0.10)]

    list_columns = list(dataset_sensors.columns)

    dataset_sensors = dataset_sensors[list_columns]
    dataset_sensors = dataset_sensors.drop(['timestamp', 'target'], axis=1)

    total_list = []

    for z in list(dataset_sensors.columns):
        total_list.append(dataset_sensors[z].var(skipna=True))

    return np.quantile(total_list, 0.05)


def columns_generic_transformation(x):
    timestamp = transform_to_timestamp_date(x["Timestamp"])
    x["timestamp"] = timestamp
    return x


def transform_to_timestamp_date(s):
    return time.mktime(datetime.strptime(s, "%m/%d/%Y %H:%M").timetuple())


def prepare_dataset(path):
    dataset = pd.read_csv(path, engine="pyarrow")
    dataset = dataset.apply(lambda x: columns_generic_transformation(x), axis=1)
    dataset = dataset.sort_values(by=['timestamp'], ascending=True)
    dataset.reset_index(drop=True, inplace=True)

    dataset = dataset.loc[(dataset['Irritable towards baby & partner'] != "")
                          & (dataset['Problems concentrating or making decision'] != "")
                          & (dataset['Feeling of guilt'] != "")]

    dataset["Feeling anxious"] = pd.Series(np.where(dataset["Feeling anxious"].values == 'Yes', 1, 0),
                                           dataset.index)
    dataset = dataset[['timestamp', 'Age', 'Feeling sad or Tearful',
                       'Irritable towards baby & partner', 'Trouble sleeping at night',
                       'Problems concentrating or making decision',
                       'Overeating or loss of appetite', 'Feeling of guilt',
                       'Problems of bonding with baby', 'Suicide attempt', 'Feeling anxious']]
    dataset = pd.get_dummies(data=dataset, columns=['Age', 'Feeling sad or Tearful',
                                                    'Irritable towards baby & partner', 'Trouble sleeping at night',
                                                    'Problems concentrating or making decision',
                                                    'Overeating or loss of appetite', 'Feeling of guilt',
                                                    'Problems of bonding with baby', 'Suicide attempt'])
    dataset = dataset.rename(columns={'Feeling anxious': 'target'})

    dataset.to_csv("datasets/dataset_processed.csv", index=False, header=True)


if __name__ == '__main__':
    prepare_dataset("datasets/post_birth.csv")
    calc_variance_threshold("datasets/dataset_processed.csv")