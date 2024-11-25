from copy import deepcopy
from typing import List, Any, Dict

import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
import hdbscan
from defenses.fedavg import FedAvg
from scipy.fftpack import dct, idct

import os


logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FreqFed(FedAvg):
    lamda: float = 0.001


    def filtering(self, temp):
        size = temp.size
        temp[np.int64(np.floor(size/2))+1:] = 0

        return temp

    def aggr(self, weight_accumulator, global_model):
        # Collecting updates
        layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'
        local_params = []
        ed = []
        for i in range(self.params.fl_no_models):
            updates_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(updates_name)
            local_model = deepcopy(global_model)


            for name, data in loaded_params.items():
                if self.check_ignored_weights(name):
                    continue
                local_model.state_dict()[name].add_(data)
                if layer_name in name:
                    temp = local_model.state_dict()[name].cpu()                
                    local_params = np.append(local_params, temp)


            print("***********************************************************")
            print(loaded_params)

            local_params = dct(local_params)

            print("----------------------------------------------------------")

            print(local_params)


            local_params = self.filtering(local_params)

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(local_params)
        
            ed = np.append(ed, self.get_update_norm(loaded_params))       
        logger.warning("FreqFed: Finish DCT transform and filtering")


        # HDBSCAN clustering
        cd = smp.cosine_distances(local_params.reshape(self.params.fl_no_models, -1))
        # logger.info(f'HDBSCAN {cd}')
        cluster = hdbscan.HDBSCAN(min_cluster_size = 
                int(self.params.fl_no_models/2+1), 
                min_samples=1, # gen_min_span_tree=True, 
                allow_single_cluster=True, metric='precomputed').fit(cd)

        cluster_labels = (cluster.labels_).tolist()
        logger.warning(f"FreqFed: cluster results {cluster_labels}")

        for i in range(self.params.fl_no_models):
            if cluster_labels[i] == -1:
                continue            
            self.accumulate_weights(weight_accumulator, loaded_params)

        return weight_accumulator