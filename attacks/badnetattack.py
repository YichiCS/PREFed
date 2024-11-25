import os
import torch
from attacks.attack import Attack

class BadnetAttack(Attack):

    def __init__(self, params, synthesizer):
        super().__init__(params, synthesizer)
        self.fixed_scales = {'normal':0.5, 
                            'backdoor':0.5}
        
        self.collusion = True
        
        self.args = self.params.args
        self.group = synthesizer.group
        self.att_ids = [g[0] for g in self.group]



    def perform_attack(self, global_model, epoch):
        
        if self.params.fl_number_of_adversaries <= 0 or \
            epoch not in range(self.params.poison_epoch,\
            self.params.poison_epoch_stop):
            return
        
        # save updates by groups
        for g in self.group:
            folder_name = os.path.join(self.params.folder_path, 'saved_updates')
            file_name = os.path.join(folder_name, f'update_{g[0]}.pth')
            loaded_params = torch.load(file_name)
            self.scale_update(loaded_params, self.params.fl_weight_scale)
            for _g in g:
                file_name = f'{folder_name}/update_{_g}.pth'
                torch.save(loaded_params, file_name)
            
        return