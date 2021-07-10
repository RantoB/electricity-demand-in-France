import pandas as pd
import numpy as np
import requests
from datetime import datetime as dt
import sys
sys.path.append('../app/')
from make_predictions import make_predictions

# Load
def loading_data(url: str):
    response = requests.get(url)

    if response.status_code == 200:
        print(f'Data from {url} successfully loaded')

    return pd.DataFrame(response.json())

# Preprocessing raw data
def preprocessing(df: pd.core.frame.DataFrame):
    # Recast date time to datetime type
    df['date_heure'] = pd.to_datetime(df['date_heure'])
    df.index = df['date_heure']
    df['day'] = df['date_heure'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['week_day'] = df['date_heure'].apply(lambda x: x.strftime('%A'))
    df['week_number'] = df['date_heure'].apply(lambda x: x.strftime('%W'))
    df['minutes'] = df['date_heure'].apply(lambda x: x.strftime('%M'))

    # Sort data by datetime
    df = df.sort_index()

    # Remove useless columns and lines full of NA (futur lines)
    col_to_drop = ['ech_comm_angleterre', 'ech_comm_espagne',
                   'ech_comm_italie', 'ech_comm_suisse', 'ech_comm_allemagne_belgique',
                  'perimetre', 'nature', 'date', 'heure']

    df = df.drop(col_to_drop, axis=1)
    df = df.dropna()

    # Reorder the columns to have the new datetime columns in first
    col_reindex =  np.array(df.columns)
    col_reindex = np.concatenate((col_reindex[-5:], col_reindex[:-5]), axis=0)

    df = df.reindex(columns=col_reindex)
    return df

# Set 'type of day' at -1 for free days
def free_day(df):
    FREE_DAYS = ['2020-01-01', '2020-04-13', '2020-05-01',
    '2020-05-08', '2020-05-10', '2020-05-21', '2020-06-01',
    '2020-07-14', '2020-08-15', '2020-11-01', '2020-11-11', '2020-12-25']

    if type(df) == pd.core.series.Series:
        for date in FREE_DAYS:
            df[df.astype(str).str.contains(date)] = -1

    elif type(df) == pd.core.frame.DataFrame:
        for date in FREE_DAYS:
            df['type_of_day'][df['date_heure'].astype(str).str.contains(date)] = -1

    return df

# Filtering data
def filter_data(df: pd.core.frame.DataFrame):
    # Transform to hourly dataframe
    df_filtered = df[df['minutes'] == '00']
    df_filtered = df_filtered.drop(['minutes'], axis=1)

    # Add an hourly frequency to the index and forward filling if some hours are missing
    df_filtered = df_filtered.asfreq('H', method='ffill')

    # Remove all columns except time and conumption columns
    df_filtered = df_filtered[['date_heure', 'day', 'consommation', 'week_day', 'week_number']]

    # Remove values for the current day
    df_filtered = df_filtered[~df_filtered['date_heure'].astype(str).str.contains(dt.now().strftime('%Y-%m-%d'))]

    # Type of day
    # Weekday : 1
    # Saturday : 0
    # Free days and Sunday : -1
    df_filtered['type_of_day'] = df_filtered['week_day']\
        .apply(lambda x: -1 if x == 'Sunday' else(0 if x == 'Saturday' else 1))

    free_day(df_filtered)

    return df_filtered

# Get train data
def train_data(df: pd.core.frame.DataFrame):
    train_mean_consumption = df['consommation'].mean()
    train_days = df['consommation'] / train_mean_consumption
    exog_train = df['type_of_day']
    return (train_days, exog_train)

# Daily data
def daily_data(df):
    daily_consumption = df.groupby('day')[['consommation', 'type_of_day']].mean()
    daily_consumption['consommation'] /= df['consommation'].mean()
    daily_consumption.index = pd.to_datetime(daily_consumption.index)
    daily_consumption.index.freq = 'D'
    return daily_consumption

# Filter the data for a given period of time
def filter_period(df, start_day, duration_days=None, duration_hours=None):
    '''
    will return df from start_day until then end of duration_days or duration_hours

    params:
    df: dataframe with datetime sort_index
    start_day: string day in format 'YYYY-mm-dd'
    duration_days or duration_hours : integer

    return
    dataframe
    '''
    if duration_hours == None:
        duration_hours = duration_days * 24

    period = pd.date_range(start=start_day, periods=duration_hours, freq='H', normalize=True, tz='UTC')

    return df[(df.index >= period[0]) & (df.index <= period[-1])]

# From the main dataframe, convert them to slices of a given duration period
def data_slices_to_list_of_dict(df, duration_days=None, duration_hours=None):
    if duration_hours == None:
        duration_hours = duration_days * 24

    duration_hours = 240
    data_list_of_dict = list()

    end_idx = int(duration_hours / 24 + 1)

    for start_days in df['day'].unique()[:-end_idx]:
        tmp_dict = {'name': start_days + '_' + str(duration_hours) +'h',
         'start_day': start_days,
         'duration': duration_hours,
         'data': filter_period(df, start_days, duration_hours=duration_hours)}

        data_list_of_dict.append(tmp_dict)

    return data_list_of_dict

def get_prediction_evaluation_follow_up(data_list_of_dict):
        pred_list = list()
        base_path = 'models/models-and-data-for-evaluation-follow-up/'

        for data_slice in data_list_of_dict:
            try:
                daily_consumption_path = base_path + f'daily_consumption_{data_slice["name"]}.pkl'
                exog_train_path = base_path + f'exog_train_{data_slice["name"]}.pkl'
                df_light_path = base_path + f'df_light_{data_slice["name"]}.pkl'
                daily_model_path = base_path + f'daily_model_{data_slice["name"]}.pkl'
                hourly_model_path = base_path + f'hourly_model_{data_slice["name"]}.pkl'
                n_pred = 24

                pred_list.append(make_predictions(daily_consumption_path,
                    exog_train_path,
                    df_light_path,
                    daily_model_path,
                    hourly_model_path,
                    n_pred)[0])

            except:
                continue

        return pred_list
