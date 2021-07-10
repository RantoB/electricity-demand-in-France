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

# Make data slices of 10 sliding days for model evaluation
data_list_of_dict = data_slices_to_list_of_dict(df_light, duration_hours=240)

if __name__  == '__main__':

    with open('../app/data/data_list_of_dict.pkl', "wb") as f:
        pickle.dump(data_list_of_dict, f)

    base_path = 'models/models-and-data-for-evaluation-follow-up/'

    for i, data_slice in enumerate(data_list_of_dict):

        try:
            train_days, exog_train = train_data(data_slice['data'])
            daily_consumption = daily_data(data_slice['data'])

            hourly_model = SARIMAX(train_days.iloc[-240:], order=(10, 1, 1), seasonal_order=(2, 0, 2, 24), exog=exog_train.iloc[-240:])
            hourly_model_fit = hourly_model.fit(disp=False)
            print('\nHourly model fitted successfully\n')

            daily_model = SARIMAX(daily_consumption['consommation'], order=(3, 1, 1), seasonal_order=(3, 2, 3,7), exog=daily_consumption['type_of_day'])
            daily_model_fit = daily_model.fit(disp=False)
            print('\nDaily model fitted successfully\n')

            with open(base_path + f'hourly_model_{data_slice["name"]}.pkl', "wb") as f:
                pickle.dump(hourly_model_fit, f)
            print('\nHourly model saved successfully.')

            with open(base_path + f'daily_model_{data_slice["name"]}.pkl', "wb") as f:
                pickle.dump(daily_model_fit, f)
            print('\nDaily model saved successfully.')

            with open(base_path + f'exog_train_{data_slice["name"]}.pkl', "wb") as f:
                pickle.dump(exog_train, f)

            with open(base_path + f'df_light_{data_slice["name"]}.pkl', "wb") as f:
                pickle.dump(df_light, f)

            with open(base_path + f'daily_consumption_{data_slice["name"]}.pkl', "wb") as f:
                pickle.dump(daily_consumption, f)

        except:
            print(f'! ! ! ! Error with {data_slice["name"]} ! ! ! !')
            continue

        print('\nMake evaluation model, step : ' + str(i) + '/' + str(len(data_list_of_dict)) + '\n')

    prediction_evaluation_follow_up = get_prediction_evaluation_follow_up(data_list_of_dict)

    with open('../app/data/prediction_evaluation_follow_up.pkl', "wb") as f:
        pickle.dump(prediction_evaluation_follow_up, f)

    print(dt.now() - start_calc)
