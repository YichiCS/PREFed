import torch

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

class DBASynthesizer(Synthesizer):

    def __init__(self, task: Task):
        super().__init__(task)
        
        self.args = self.params.args
        self.dba = self.args.dba
        if self.dba:   
            self.args.groups = 4
        self.shift = 0
        
        self.group = [[] for _ in range(self.args.groups)]
        for i in range(self.params.fl_number_of_adversaries):
            self.group[i%self.args.groups].append(i)
        print(f'Attacker Groups: {self.group}')
        
        self.trigger = [
        torch.tensor([
            [0., 0., 1.], 
            [0., 1., 0.], 
            [1., 0., 1.], 
        ]), 
        torch.tensor([
            [1., 0., 0.], 
            [0., 1., 0.], 
            [1., 0., 1.], 
        ]), 
        torch.tensor([
            [1., 0., 1.], 
            [0., 1., 0.], 
            [1., 0., 0.], 
        ]), 
        torch.tensor([
            [1., 0., 1.], 
            [0., 1., 0.], 
            [0., 0., 1.], 
        ])]
            
        # self.label = self.params.backdoor_label
        self.label = [self.params.backdoor_label for _ in range(self.args.groups)]
        self.task = task 
        
    def synthesize_inputs(self, batch, attack_portion=None, index=None, test=False):
        
        group_id = next(i for i, idx in enumerate(self.group) if index in idx)  
        batch.inputs[:attack_portion] = self.task.denormalize(batch.inputs[:attack_portion])
        
        if not test:
            trigger = self.trigger[group_id]
            _x, _y = trigger.shape
            x = 5 * (group_id // 2)
            y = 5 * (group_id % 2)
            batch.inputs[:attack_portion, :, x:x+_x, y:y+_y] = trigger
        elif test:
            for i in range(self.args.groups):
                _x, _y = self.trigger[i].shape
                x = 5 * (i // 2) + self.shift
                y = 5 * (i % 2) + self.shift
                
                batch.inputs[:attack_portion, :, x:x+_x, y:y+_x] = self.trigger[i]
            
        batch.inputs[:attack_portion] = self.task.normalize(batch.inputs[:attack_portion])

        return
    

    def synthesize_labels(self, batch, attack_portion=None, index=None, test=False):
        
        group_id = next(i for i, idx in enumerate(self.group) if index in idx) 
        label = self.label[group_id] 
        
        batch.labels[:attack_portion].fill_(label)

        return
    
    
    
    
    