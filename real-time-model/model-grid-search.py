import os
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

records_path = os.path.join('models', 'records.txt')
path_to_df = os.path.join('..', 'processed-data', 'hourly_elec_prod_may.df')
with open(path_to_df, 'rb') as f:
    df = pickle.load(f)

# path_to_old_df = os.path.join('..', 'processed-data', 'real_time_elec_production_may.df')
# with open(path_to_old_df, 'rb') as f:
#     old_df = pickle.load(f)
# old_test_day = old_df['Consommation (MW)'].loc['2020-05-07 00:00:00+02:00':'2020-05-07 23:45:00+02:00']

train_start = '2020-05-10 00:00:00+02:00'
train_end = '2020-05-16 23:00:00+02:00'
test_start = '2020-05-17 00:00:00+02:00'
test_end = '2020-05-17 23:00:00+02:00'

train_days = df['Consommation (MW)'].loc[train_start:train_end]
test_day = df['Consommation (MW)'].loc[test_start:test_end]
exog_train = df[['TMoy (°C)', 'type_of_day']].loc[train_start:train_end]
exog_test = df[['TMoy (°C)', 'type_of_day']].loc[test_start:test_end]

train_days_rescaled = train_days / train_days.mean()
test_day_rescaled = test_day / train_days.mean()
exog_train['TMoy (°C)-rescaled'] = exog_train['TMoy (°C)'] / exog_train['TMoy (°C)'].mean()
exog_test['TMoy (°C)-rescaled'] = exog_test['TMoy (°C)'] / exog_train['TMoy (°C)'].mean()

AR = [9, 10]
MA = [1, 2]
seas_P = [1]
seas_D = [1]
seas_Q = [1, 2]

def generate_sarima_models(train, train_exog, test_exog, AR, MA, seas_P, seas_D, seas_Q):
    for p in AR:
        for q in MA:
            for P in seas_P:
                for D in seas_D:
                    for Q in seas_Q:
                        p = p
                        d = 1
                        q = q
                        P = P
                        D = D
                        Q = Q
                        s = 24

                        with open (records_path, 'a') as f:
                            start_time = datetime.now()
                            f.write(f"\n{start_time.strftime('%Y-%m-%d - %H:%M:%S')}\nSARIMA_order=({p}-{d}-{q})_seasonal_order=({P}-{D}-{Q}-{s})")

                        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s), exog=exog_train)
                        model_fit = model.fit(disp=False)
                        prediction = model_fit.get_prediction(start=test_start, end=test_end, exog=exog_test)

                        model_to_save = {'model_name': f"SARIMA_or=({p}-{d}-{q})_seor=({P}-{D}-{Q}-{s}_exog_scaled)",
                                      'p': p,
                                      'd': d,
                                      'q': q,
                                      'P': P,
                                      'D': D,
                                      'Q': Q,
                                      's': s,
                                      'model': model,
                                      'model_fitted': model_fit,
                                      'prediction': prediction}

                        calc_time = datetime.now() - start_time
                        with open (records_path, 'a') as f:
                            f.write(f"\ndelta time : {str(calc_time)}\n--------------------------")

                        path_to_save_model = os.path.join('models', f'{model_to_save["model_name"]}.24h')
                        with open(path_to_save_model, 'wb') as f:
                            pickle.dump(model_to_save, f)

if __name__ == '__main__':
    generate_sarima_models(train_days_rescaled, exog_train[['TMoy (°C)-rescaled', 'type_of_day']], exog_test[['TMoy (°C)-rescaled', 'type_of_day']], AR, MA, seas_P, seas_D, seas_Q)
