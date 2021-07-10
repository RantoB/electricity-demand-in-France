from datetime import datetime as dt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from utils import *

start_calc = dt.now()

BASE_URL_pre = 'https://opendata.reseaux-energies.fr/api/v2/catalog/datasets/'
BASE_URL_port = '/exports/json?rows=-1&pretty=false&timezone=UTC'
real_time_elec = 'eco2mix-national-tr'
temperature = 'temperature-quotidienne-regionale'

# Loading
df = loading_data(BASE_URL_pre + real_time_elec + BASE_URL_port)
# temp_df = loading_data(BASE_URL_pre + temperature + BASE_URL_port)

df = preprocessing(df)

# Filtered data: remove data not used
df_light = filter_data(df)

# Endogenous data for hourly predictions
train_days, exog_train = train_data(df_light)

# Data for daily model
daily_consumption = daily_data(df_light)

if __name__  == '__main__':
    try:
        hourly_model = SARIMAX(train_days.iloc[-240:], order=(10, 1, 1), seasonal_order=(2, 0, 2, 24), exog=exog_train.iloc[-240:])
        hourly_model_fit = hourly_model.fit(disp=False)
        print('\nHourly model fitted successfully\n')

        daily_model = SARIMAX(daily_consumption['consommation'], order=(3, 1, 1), seasonal_order=(3, 2, 3,7), exog=daily_consumption['type_of_day'])
        daily_model_fit = daily_model.fit(disp=False)
        print('\nDaily model fitted successfully\n')
    except:
        print('\nModel not launched')

    try:
        with open('../app/prediction-models/hourly_model.pkl', "wb") as f:
            pickle.dump(hourly_model_fit, f)
        print('\nHourly model saved successfully.')

        with open('../app/prediction-models/daily_model.pkl', "wb") as f:
            pickle.dump(daily_model_fit, f)
        print('\nDaily model saved successfully.')

        with open('../app/data/exog_train.pkl', "wb") as f:
            pickle.dump(exog_train, f)

        with open('../app/data/df_light.pkl', "wb") as f:
            pickle.dump(df_light, f)

        with open('../app/data/daily_consumption.pkl', "wb") as f:
            pickle.dump(daily_consumption, f)

        with open('../app/data/date.pkl', "wb") as f:
            pickle.dump(start_calc, f)

    except:
        print('Predictions not saved')

    print(dt.now() - start_calc)
