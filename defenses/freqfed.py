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

class FreqFed(FedAvg):
    lamda: float = 0.001

    def filtering(self, temp):
        # 计算过滤后数组的长度，即原数组长度的一半（使用整除保证结果为整数）
        filtered_length = len(temp) // 2
        # 截取前半部分元素，使用max确保不会低于0
        F = temp[:filtered_length]
        # 返回过滤后的数组
        return F

    def cluster(self, F_list):
        distances_matrix = np.zeros((len(F_list), len(F_list)))
        for i in range(len(F_list)):
            for j in range(i + 1, len(F_list)):
                distances_matrix[i, j] = 1 - cosine_similarity(F_list[i].reshape(1, -1), F_list[j].reshape(1, -1))[0][0]
                distances_matrix[j, i] = distances_matrix[i, j]
        
        cluster = hdbscan.HDBSCAN(min_cluster_size=int(self.params.fl_no_models/2+1), 
                                  allow_single_cluster=True, 
                                  metric='precomputed').fit(distances_matrix)
        return cluster.labels_.tolist()

    def aggr(self, weight_accumulator, global_model):
        # Collecting updates
        local_params_list = []

        for i in range(self.params.fl_no_models):
            updates_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(updates_name)
            

            local_params = np.append(loaded_params['fc.weight'].cpu().numpy(),loaded_params['fc.bias'].cpu().numpy())

            local_params = dct(local_params)

            # why after dct, the local_params includ Nan?
            # if np.isnan(local_params).any():
            #     import pdb
            #     pdb.set_trace()


            filtered_params = self.filtering(local_params)
            local_params_list.append(filtered_params.flatten())


        logger.warning("FreqFed: Finish DCT transform and filtering")

        try:
            cluster_labels = self.cluster(local_params_list)
        except:
            # import pdb
            # pdb.set_trace()
            cluster_labels = self.cluster(local_params_list)
        
        logger.warning(f"FreqFed: cluster results {cluster_labels}")


        # Aggregate weights from models in the majority cluster
        for i in range(self.params.fl_no_models):
            if cluster_labels[i] == -1:
                continue

            update_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(update_name)
            self.accumulate_weights(weight_accumulator, loaded_params)

        return weight_accumulator
