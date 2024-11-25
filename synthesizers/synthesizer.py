from tasks.batch import Batch
from tasks.task import Task
from utils.parameters import Params


class Synthesizer:
    params: Params
    task: Task

    def __init__(self, task: Task):
        self.task = task
        self.params = task.params
        self.args = self.params.args
        
        self.dba = False
        
        self.group = [[] for _ in range(self.args.groups)]
        for i in range(self.params.fl_number_of_adversaries):
            self.group[i%self.args.groups].append(i)


    def make_backdoor_batch(self, batch: Batch, test=False, attack=True, index=None) -> Batch:

        # Don't attack if only normal loss task.
        if not attack:
            return batch

        if test:
            attack_portion = batch.batch_size
        else:
            attack_portion = round(
                batch.batch_size * self.params.poisoning_proportion)

        backdoored_batch = batch.clone()
        self.apply_backdoor(backdoored_batch, attack_portion, index, test)

        return backdoored_batch

    def apply_backdoor(self, batch, attack_portion, index=None, test=False):
        """
        Modifies only a portion of the batch (represents batch poisoning).

        :param batch:
        :return:
        """
        self.synthesize_inputs(batch=batch, attack_portion=attack_portion, index=index, test=test)
        self.synthesize_labels(batch=batch, attack_portion=attack_portion, index=index, test=test)

        return

    def synthesize_inputs(self, batch, attack_portion=None, index=None, test=False):
        raise NotImplemented

    def synthesize_labels(self, batch, attack_portion=None, index=None, test=False):
        raise NotImplemented
