
import os
import torch
from attacks.attack import Attack
from attacks.loss_functions import compute_cos_sim_loss
import sklearn.metrics.pairwise as smp
import numpy as np
from torchvision.utils import save_image
from copy import deepcopy
from tqdm import tqdm


import sys

#sys.stdout = open('1_cd_output.txt', 'w')



class OptAttack(Attack):
    def __init__(self, params, synthesizer):
        super().__init__(params, synthesizer)
        self.loss_tasks.append('cs_constraint')
        self.fixed_scales = {'normal':0.0, 
                            'backdoor':0.5, 
                            'cs_constraint':0.5}
        
        self.trigger_optimize = True
        self.trigger_init = True
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
            # self.scale_update(loaded_params, self.params.fl_weight_scale)
            for _g in g:
                file_name = f'{folder_name}/update_{_g}.pth'
                torch.save(loaded_params, file_name)
            
        return
    
    def optimize_trigger(self, local_model, user, hlpr, round_participants, info=True):
        
        group_id = next(i for i, idx in enumerate(self.group) if user.user_id in idx)
        if info:
            print(f'Group {group_id} Attacking Label {hlpr.synthesizer.label[group_id]}')
        
        global_model = deepcopy(local_model)
        clean_models, trigger_models = [], []
        attackers = []
        
        for i in self.group[group_id]:
            for u in round_participants:
                if u.user_id == i:
                    attackers.append(u)
        
        for att in attackers:
            clean_model, trigger_model = self.get_user_model(local_model, hlpr, att)
            clean_models.append(clean_model)
            trigger_models.append(trigger_model)
        
        args = self.params.args
        criterion = hlpr.task.criterion  
        # trigger optimize
        if args.name == 'cx_method':
            
            for att, clean_model, trigger_model in zip(attackers, clean_models, trigger_models):
                
                cos_sim_loss = self.get_cos_sim_loss(hlpr, clean_model, trigger_model, global_model)
                trigger_model.eval()
                
                self.synthesizer.trigger[group_id] = torch.autograd.Variable(self.synthesizer.trigger[group_id], requires_grad=True)
                trigger_optimizer = torch.optim.RAdam(params=[self.synthesizer.trigger[group_id]], lr=args.lr, weight_decay=args.weight_decay)
                
                beta = args.beta
                
                for i, data in enumerate(att.train_loader):
                    batch = hlpr.task.get_batch(i, data)
                    batch_back = hlpr.attack.synthesizer.make_backdoor_batch(batch, attack=True, index=att.user_id)
                    logits = trigger_model.forward(batch_back.inputs)
                    loss = criterion(logits, batch_back.labels)
                    loss_regu = torch.mean(loss)
                    
                    loss_regu.data = beta * cos_sim_loss + (1 - beta) * loss_regu.data
                    
                    trigger_optimizer.zero_grad()
                    loss_regu.backward(retain_graph=True)
                    trigger_optimizer.step()
                
                

                self.synthesizer.trigger[group_id] = torch.clamp(self.synthesizer.trigger[group_id].data, -1, 1)
        
        else:
            raise NotImplementedError
                    
        return clean_model
    
    def get_cos_sim_loss(self, hlpr, clean_model, trigger_model, global_model):
        
        local_params = []
    
        clean_update = hlpr.attack.get_fl_update(clean_model, global_model)
        trigger_update = hlpr.attack.get_fl_update(trigger_model, global_model)
        
        for update in [clean_update, trigger_update]:
            for name, _ in update.items():
                if 'fc' in name:
                    temp = update[name].cpu().numpy()
                    local_params = np.append(local_params, temp) 
        

        cosine_dist = smp.cosine_distances(local_params.reshape(2,-1))[0][1]
        cos_sim_loss = 1 - torch.tensor(cosine_dist)
        
        return cos_sim_loss
    def get_user_model(self, local_model, hlpr, user):
        
        global_model = deepcopy(local_model)
        clean_model = deepcopy(local_model)
        trigger_model = deepcopy(local_model)
    
        criterion = hlpr.task.criterion
        optimizer_clean = hlpr.task.make_optimizer(clean_model)
        optimizer_trigger = hlpr.task.make_optimizer(trigger_model)   
        
        # train local model on clean data
        for _ in range(hlpr.params.fl_poison_epochs):
            clean_model = self.multi_loss_train(hlpr, clean_model, optimizer_clean, criterion, user, attack=False, global_model=global_model)
        # backdoored local model on poisoned data
        for _ in range(hlpr.params.fl_poison_epochs):
            trigger_model = self.multi_loss_train(hlpr, trigger_model, optimizer_trigger, criterion, user, attack=True, global_model=global_model)
            
        return clean_model, trigger_model
    

    def multi_loss_train(self, hlpr, model, optimizer, criterion, user, attack=True, global_model=None):
    
        index = user.user_id
        train_loader = user.train_loader
        
        model.train()
        for i, data in enumerate(train_loader):
            batch = hlpr.task.get_batch(i, data)
            model.zero_grad()
            loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, global_model, index=index)
            loss.backward()
            optimizer.step()
            if i == hlpr.params.max_batch_id:
                break
            
        return model
    
    
    def init_trigger(self, hlpr):
        
        args = self.args
            
        print(f'Trigger Init Method: {self.args.init_mode}')
        
        if self.args.init_mode == 'zero':
            save_image(hlpr.attack.synthesizer.trigger, os.path.join(hlpr.params.folder_path, 'triggers', f'_init_triggers.png'))
        elif self.args.init_mode == 'offline':
            
            local_model = hlpr.task.build_model()
            ckpt_path = os.path.join('saved_models', f"{self.args.init_model}")
            loaded_params = torch.load(ckpt_path, map_location=torch.device('cpu'))
            local_model.load_state_dict(loaded_params['state_dict'])
            local_model = local_model.cuda()
            
            # obtain attacker informations
            round_participants = hlpr.task.sample_users_for_round(self.params.poison_epoch)
                
            for g in self.group:
                
                for u in round_participants:
                    if u.user_id == g[0]:
                        user = u
                    
                for _ in tqdm(range(args.init_epoch)):
                    
                    self.optimize_trigger(local_model, user, hlpr, round_participants, info=False)
                    
            
            save_image(hlpr.attack.synthesizer.trigger, os.path.join(hlpr.params.folder_path, 'triggers', f'_init_triggers.png'))
            
        else:
            raise ValueError('Wrong Value of Trigger Init Method')
        
        return 