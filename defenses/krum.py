
from copy import deepcopy
from typing import List, Any, Dict
from collections import defaultdict

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

class Krum(FedAvg):

    # there are 20个
    
    # Median aggregation
    def aggr(self, weight_accumulator, model):
        # Initialize a dictionary to hold the list of weights for each parameter
        weights_list = {name: [] for name in model.state_dict().keys()}

        distances = defaultdict(dict)
        non_malicious_count = 16


        # Load weights from each file and append to the list
        for i in range(self.params.fl_no_models):
            updates_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(updates_name, map_location=self.params.device)
            for key, value in loaded_params.items():
                if key in weights_list:
                    weights_list[key].append(value.cpu().numpy())

        num = 0
        for key in weights_list.keys():
            if num == 0:
                for i in range(len(weights_list[key])):
                    for j in range(i):
                        distances[i][j]= distances[j][i] = np.linalg.norm(weights_list[key][i]-weights_list[key][j])

                num = 1
            else:
                for i in range(len(weights_list[key])):
                    for j in range(i):
                        
                        distances[j][i] += np.linalg.norm(weights_list[key][i]- weights_list[key][j])
                        
                        distances[i][j] += distances[j][i]

        minimal_error = 1e20
        for user in distances.keys():
              errors = sorted(distances[user].values())
              current_error = sum(errors[:non_malicious_count])
              
              if current_error < minimal_error:
                  minimal_error = current_error
                  minimal_error_index = user

        for key in weights_list.keys():
            
            if 'num_batches_tracked' in key:
                continue

            weights_list[key] = torch.tensor(weights_list[key][minimal_error_index])
                

        # Update the model with the median weight
        self.accumulate_weights(weight_accumulator, \
                {key:torch.tensor(weights_list[key]).to(self.params.device) for key in weights_list} )



