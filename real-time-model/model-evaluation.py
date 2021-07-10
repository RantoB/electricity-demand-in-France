import os
import pickle
from sklearn.metrics import mean_squared_error
from models import test_day, train_days
# from models import old_test_day
from statsmodels.tsa.statespace.sarimax import SARIMAX

path_to_score_txt = 'score.csv'
path_to_save_list = os.path.join('models', '24h-prediction-list')
path_to_models = os.path.join('models')

# with open (path_to_score_txt, 'w') as f:
#     f.write("\nmodel_name,mse_score")

with open(path_to_save_list, 'rb') as f:
    models = pickle.load(f)

# models = dict()

for pkl in os.listdir(path_to_models):
    if pkl.endswith(".24h"):
        print(pkl)
        try:
            file = os.path.join(path_to_models, pkl)
            with open(file, 'rb') as f:
                model = pickle.load(f)

            pred = model['prediction'].predicted_mean * train_days.mean()

            if len(pred) == 24:
                models[model['model_name']] = pred

                mse = mean_squared_error(test_day, pred)

                with open (path_to_score_txt, 'a') as f:
                    f.write(f"\n{model['model_name']}-10:16-05, {mse}")

            else:
                continue
                # models[model['model_name']] = pred
                #
                # mse = mean_squared_error(old_test_day, pred)
                #
                # with open (path_to_score_txt, 'a') as f:
                #     f.write(f"\n{model['model_name']}, {mse}")

        except:
            print(f"Problem with {pkl} ...")
    else:
        continue


with open(path_to_save_list, 'wb') as f:
    pickle.dump(models, f)
