from copy import deepcopy
from typing import List, Any, Dict

import torch
import logging
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from defenses.fedavg import FedAvg
from scipy.fftpack import dct, idct

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class trimmed_Mean(FedAvg):
    
    # Median aggregation
    def aggr(self, weight_accumulator, model):
        # Initialize a dictionary to hold the list of weights for each parameter
        weights_list = {name: [] for name in model.state_dict().keys()}


        # Load weights from each file and append to the list
        for i in range(self.params.fl_no_models):
            updates_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(updates_name, map_location=self.params.device)
            for key, value in loaded_params.items():
                if key in weights_list:
                    weights_list[key].append(value.cpu().numpy())

                    


        # Calculate the median weight for each parameter
        median_weights = {name: np.median(np.array(values), axis=0) for name, values in weights_list.items()}

        # Update the model with the median weight
        self.accumulate_weights(weight_accumulator, \
                {key:torch.tensor(median_weights[key]).to(self.params.device) for key in median_weights} )
        
        for key in weights_list.keys():

            tmp = np.array(weights_list[key])
            med = np.median(tmp, axis=0)
            new_tmp = []
            for i in range(len(tmp)): # cal each client (weights - median)
                new_tmp = np.array(tmp[i]-med)
            
            new_tmp= np.array(new_tmp)
            good_vals = np.argsort(abs(new_tmp), axis=0)
            good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
            k_weight = np.array(np.mean(good_vals) + med)
            weights_list[key] = torch.from_numpy(k_weight).to(self.params.device)
        
        self.accumulate_weights(weight_accumulator, \
                {key:weights_list[key].to(self.params.device) for \
                    key in weights_list})
