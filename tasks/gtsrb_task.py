import random

import numpy as np

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import Subset

from models.resnet_gtsrb import resnet18
from tasks.task import Task
import os


class GTSRBTask(Task):

    normalize = transforms.Normalize((0.3337, 0.3064, 0.3171), 
                                     (0.2672, 0.2564, 0.2629))
    
    denormalize = transforms.Normalize(
        (-0.3337/0.2672, -0.3664/0.2564, -0.3171/0.2629),
        (1/0.2672, 1/0.2564, 1/0.2629)
    )

    def load_data(self):
        self.load_gtsrb_data()

        if self.params.load_indices:
            indices_per_participant = np.load('./load_indices/gtsrb_indices_per_participant.npy', allow_pickle=True).item()
            
            print("Data_loader: load the train dataset indices")
            train_loaders = [self.get_train(indices) for pos, indices in
                            indices_per_participant.items()]

        elif self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution

            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)

            
            np.save('./load_indices/gtsrb_indices_per_participant', indices_per_participant)
            train_loaders = [self.get_train(indices) for pos, indices in
                             indices_per_participant.items()]
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)

            np.save('./load_indices/gtsrb_indices_per_participant', indices_per_participant)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
            
        self.fl_train_loaders = train_loaders
        return

    def load_gtsrb_data(self):

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_dataset = torchvision.datasets.GTSRB(
            root=self.params.data_path,
            split = 'train',
            transform = train_transform,)

        self.test_dataset = torchvision.datasets.GTSRB(
            root=self.params.data_path,
            split = 'test',
            transform = test_transform,)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=8, pin_memory=True)

    def build_model(self) -> None:
        return resnet18(pretrained=False)
