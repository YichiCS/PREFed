import random

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import Subset

from models.resnet_tinyimagenet import resnet18
from tasks.task import Task
import os

import numpy as np


class ImagenetTask(Task):

    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))

    denormalize = transforms.Normalize(
        (-0.485/0.229, -0.456/0.224, -0.406/0.225),
        (1/0.229, 1/0.224, 1/0.225)
    )



    def load_data(self):
        self.load_imagenet_data()

        if self.params.load_indices:
            indices_per_participant = np.load('./load_indices/imagenet_indices_per_participant.npy', allow_pickle=True).item()
            
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
            
            np.save('./load_indices/imagenet_indices_per_participant', indices_per_participant)
            train_loaders = [self.get_train(indices) for pos, indices in
                             indices_per_participant.items()]
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            split = min(self.params.fl_total_participants / 100, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        return

    def load_imagenet_data(self):

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.params.data_path, 'tiny-imagenet-200/train'),
            train_transform)

        self.test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.params.data_path, 'tiny-imagenet-200/val'),
            test_transform)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=8, pin_memory=True)


    def build_model(self) -> None:
        return resnet18(pretrained=False)
