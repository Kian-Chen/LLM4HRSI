import os.path

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.DINEOF import DINEOF
from data_provider.data_loader import Dataset_Trad


class Exp_Trad:
    def __init__(self, args):
        self.args = args
        self.dataset = Dataset_Trad(args)

    def evaluate(self, setting):
        data, missing_mask, eval_mask = self.dataset.get_data()
        
        # Instantiate DINEOF and fit data
        model = DINEOF(self.args, tensor_shape=data.shape)
        model.fit(data, missing_mask)
        
        # evaluate
        indices = np.argwhere(eval_mask == 1)
        predictions = model.predict(indices)
        
        true_values = data[eval_mask == 1]
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mae = mean_absolute_error(true_values, predictions)
        mape = np.mean(np.abs((true_values - predictions) / true_values))

        with open(self.args.log_dir + self.args.log_name, 'a') as f:
            f.write(setting + "  \n")
            print(f'{self.args.source_names[0]} - rmse:{rmse}, mae:{mae}, mape:{mape}')
            f.write(
                f'{self.args.source_names[0]} - rmse:{rmse}, mae:{mae}, mape:{mape}\n')
