import torch

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

class OptimizeSynthesizer(Synthesizer):

    def __init__(self, task: Task):
        super().__init__(task)
        
        self.args = self.params.args
        self.mode = self.args.mode
        
        self.group = [[] for _ in range(self.args.groups)]
        for i in range(self.params.fl_number_of_adversaries):
            self.group[i%self.args.groups].append(i)
        print(f'Attacker Groups: {self.group}')
        
        # TODO mask
        self.trigger = [torch.zeros(self.params.input_shape).cuda() for _ in range(self.args.groups)]
        if self.mode == 'blend':
            self.mask = (torch.ones(self.params.input_shape).cuda()) * self.args.blend_alpha 
        elif self.mode == 'patch':
            self.x, self.y = self.args.patch_x, self.args.patch_y
            self.px, self.py = self.args.patch_px, self.args.patch_py
            self.mask = (torch.zeros(self.params.input_shape).cuda())
            self.mask[:, self.px:self.px+self.x, self.py:self.py+self.y] = 1
        else:
            raise ValueError('Wrong Value of OptAttack Mode')
            
        self.label = [self.params.backdoor_label for _ in range(self.args.groups)]
        self.task = task 
        
    def synthesize_inputs(self, batch, attack_portion=None, index=None, test=False):
            
        group_id = next(i for i, idx in enumerate(self.group) if index in idx)
        trigger = self.trigger[group_id]
      
        batch.inputs[:attack_portion] = self.task.denormalize(batch.inputs[:attack_portion])
        batch.inputs[:attack_portion] = self.mask * trigger + (1 - self.mask) * batch.inputs[:attack_portion]
        batch.inputs[:attack_portion] = self.task.normalize(batch.inputs[:attack_portion])

        return
    

    def synthesize_labels(self, batch, attack_portion=None, index=None, test=False):
        
        group_id = next(i for i, idx in enumerate(self.group) if index in idx) 
        label = self.label[group_id] 
        
        batch.labels[:attack_portion].fill_(label)

        return
    
    
    
    
    