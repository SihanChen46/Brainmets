import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path
import os
import gc
import functools 
from shutil import copyfile
from tqdm import tqdm
from apex import amp
import dill

import os, glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
from torchvision import models
from torchsummary import summary

from Brainmets.Brainmets_v2.losses import *
from Brainmets.evaluation import *
from Brainmets.Brainmets_v2.model import *

from efficientnet_pytorch_3d import EfficientNet3D

class SegmentationTrainer():
    def __init__(
            self,
            name,
            model,
            train_set,
            valid_set,
            test_set,
            bs,
            lr,
            max_lr,
            loss_func,
            device):
        self.device = device
        self.name = name
        self.lr = lr
        self.bs = bs
        self.loss_function = loss_func
        self.metrics = compute_per_channel_dice
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=bs,
            shuffle=True,
            pin_memory=False)
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=bs,
            shuffle=False,
            pin_memory=False)
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=bs,
            shuffle=False,
            pin_memory=False)
        if model == 'ResidualUNet3D':
            model = ResidualUNet3D(1, 1, True).to(self.device).float()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            self.tmp_optimizer = optim.Adam(model.parameters(), lr=lr)
            self.model, self.optimizer = amp.initialize(
                model, optimizer, opt_level='O2')
        self.max_lr = max_lr
        self.lrs = []
        self.model_state_dicts = []

    def fit(self, epochs, print_each_img, use_cycle=False):
        torch.cuda.empty_cache()
        self.train_losses = []
        self.valid_losses = []
        self.train_scores = []
        self.valid_scores = []

        self.scheduler = OneCycleLR(
            self.tmp_optimizer,
            self.max_lr,
            epochs=epochs,
            steps_per_epoch=1,
            div_factor=25.0,
            final_div_factor=100)
        for epoch in range(epochs):
            self.scheduler.step()
            lr = self.tmp_optimizer.param_groups[0]['lr']
            self.lrs.append(lr)
        del self.tmp_optimizer, self.scheduler
        gc.collect()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_score = 0
            print('epoch: ' + str(epoch))
            if use_cycle:
                lr = self.lrs[epoch]
                self.optimizer.param_groups[0]['lr'] = lr
            else:
                lr = self.lr
            print(lr)
            for index, batch in tqdm(
                enumerate(
                    self.train_loader), total=len(
                    self.train_loader)):
                sample_img, sample_mask = batch
                sample_img = sample_img.to(self.device)
                sample_mask = sample_mask.to(self.device)
                predicted_mask = self.model(sample_img)
                loss = self.loss_function(predicted_mask, sample_mask)
#                 score = self.metrics(predicted_mask,sample_mask)
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
#                 total_score += score.item()
                if print_each_img:
                    print('batch loss: ' + str(loss.item()))
                del batch, sample_img, sample_mask, predicted_mask, loss, scaled_loss
                gc.collect()
                torch.cuda.empty_cache()
            print('total_loss: ' + str(total_loss / len(self.train_loader)))
            self.train_losses.append(total_loss / len(self.train_loader))
#             self.train_scores.append(total_score/len(self.train_set))
            val_score = self.val()
            self.save_checkpoint(self.name, epoch, val_score)

    def val(self):
        torch.cuda.empty_cache()
        self.model.eval()
        total_val_loss = 0
        total_val_score = 0
        for index, val_batch in tqdm(
            enumerate(
                self.valid_loader), total=len(
                self.valid_loader)):
            val_sample_img, val_sample_mask = val_batch
            val_sample_img = val_sample_img.to(self.device)
            val_sample_mask = val_sample_mask.to(self.device)
            del val_batch
            gc.collect()
            with torch.no_grad():
                val_predicted_mask = self.model(val_sample_img)
            val_loss = self.loss_function(val_predicted_mask, val_sample_mask)
            val_score = self.metrics(val_predicted_mask, val_sample_mask)
            total_val_loss += val_loss.item()
            total_val_score += val_score.item()
            del val_sample_img, val_sample_mask, val_predicted_mask, val_loss, val_score
            gc.collect()
        print('total_valid_score: ' + str(total_val_score / len(self.valid_set)))
        torch.cuda.empty_cache()
        self.valid_losses.append(total_val_loss / len(self.valid_loader))
        self.valid_scores.append(total_val_score / len(self.valid_loader))
        return total_val_score / len(self.valid_loader)

    def predict(self):
        self.model.eval()
        total_test_loss = 0
        total_test_score = 0
        for index, test_batch in tqdm(
            enumerate(
                self.test_loader), total=len(
                self.test_loader)):
            test_sample_img, test_sample_mask = test_batch
            test_sample_img = test_sample_img.to(self.device)
            test_sample_mask = test_sample_mask.to(self.device)
            del test_batch
            gc.collect()
            with torch.no_grad():
                test_predicted_mask = self.model(test_sample_img)
            test_loss = self.loss_function(
                test_predicted_mask, test_sample_mask)
            test_score = self.metrics(test_predicted_mask, test_sample_mask)
            total_test_loss += test_loss.item()
            total_test_score += test_score.item()
            del test_sample_img, test_sample_mask, test_predicted_mask, test_loss, test_score
            gc.collect()
        print('test_score: ' + str(total_test_score / len(self.test_loader)))
        torch.cuda.empty_cache()
        self.test_score = total_test_score / len(self.test_loader)
        return total_test_score / len(self.test_loader)

    def save_checkpoint(self, name, epoch, val_score):
        if not os.path.exists('./results'):
            os.mkdir('./results')
        if not os.path.exists('./results/' + name):
            os.mkdir('./results/' + name)
        dill.dump(
            self,
            open(
                './results/' +
                name +
                '/epoch_' +
                str(epoch) +
                '_val_score=' +
                str(val_score) +
                '.pkl',
                'wb'))

    @staticmethod
    def load_best_checkpoint(name):
        checkpoints = sorted([checkpoint for checkpoint in os.listdir(
            './results/' + name) if checkpoint.startswith('epoch')])
        best_epoch = np.argmax([float(checkpoint.split('=')[1].split('.')[
                               1][:10]) for checkpoint in checkpoints])
        best_epoch = int(checkpoints[best_epoch].split('_')[1])
        print('best_epoch: ', best_epoch)
        best_checkpoint = [
            checkpoint for checkpoint in checkpoints if checkpoint.startswith(
                'epoch_' + str(best_epoch))][0]
        return dill.load(
            open(
                './results/' +
                name +
                '/' +
                best_checkpoint,
                'rb'))
    
class RegressionTrainer():
    def __init__(
            self,
            name,
            model,
            train_set,
            valid_set,
            test_set,
            bs,
            lr,
            max_lr,
            loss_func,
            device):
        self.device = device
        self.name = name
        self.lr = lr
        self.bs = bs
        self.loss_function = loss_func
        self.metrics = regression_accuracy
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=bs,
            shuffle=True,
            pin_memory=False)
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=bs,
            shuffle=False,
            pin_memory=False)
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=bs,
            shuffle=False,
            pin_memory=False)
        if model == 'EfficientNet3D':
            model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 1}, in_channels=1).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            self.tmp_optimizer = optim.Adam(model.parameters(), lr=lr)
            self.model, self.optimizer = amp.initialize(
                model, optimizer, opt_level='O2')
        self.max_lr = max_lr
        self.lrs = []
        self.model_state_dicts = []

    def fit(self, epochs, print_each_img, use_cycle=False):
        torch.cuda.empty_cache()
        self.train_losses = []
        self.valid_losses = []
        self.train_scores = []
        self.valid_scores = []

        self.scheduler = OneCycleLR(
            self.tmp_optimizer,
            self.max_lr,
            epochs=epochs,
            steps_per_epoch=1,
            div_factor=25.0,
            final_div_factor=100)
        for epoch in range(epochs):
            self.scheduler.step()
            lr = self.tmp_optimizer.param_groups[0]['lr']
            self.lrs.append(lr)
        del self.tmp_optimizer, self.scheduler
        gc.collect()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_score = 0
            print('epoch: ' + str(epoch))
            if use_cycle:
                lr = self.lrs[epoch]
                self.optimizer.param_groups[0]['lr'] = lr
            else:
                lr = self.lr
            print(lr)
            for index, batch in tqdm(
                enumerate(
                    self.train_loader), total=len(
                    self.train_loader)):
                sample_img, sample_target = batch
                sample_img = sample_img.to(self.device)
                sample_target = sample_target.to(self.device)
                predicted_target = self.model(sample_img)
                loss = self.loss_function(predicted_target, sample_target)
#                 score = self.metrics(predicted_target,sample_target)
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
#                 total_score += score.item()
                if print_each_img:
                    print('batch loss: ' + str(loss.item()))
                del batch, sample_img, sample_target, predicted_target, loss, scaled_loss
                gc.collect()
                torch.cuda.empty_cache()
            print('total_loss: ' + str(total_loss / len(self.train_loader)))
            self.train_losses.append(total_loss / len(self.train_loader))
#             self.train_scores.append(total_score/len(self.train_set))
            val_score = self.val()
            self.save_checkpoint(self.name, epoch, val_score)

    def val(self):
        torch.cuda.empty_cache()
        self.model.eval()
        total_val_loss = 0
        total_val_score = 0
        for index, val_batch in tqdm(
            enumerate(
                self.valid_loader), total=len(
                self.valid_loader)):
            val_sample_img, val_sample_target = val_batch
            val_sample_img = val_sample_img.to(self.device)
            val_sample_target = val_sample_target.to(self.device)
            del val_batch
            gc.collect()
            with torch.no_grad():
                val_predicted_target = self.model(val_sample_img)
            val_loss = self.loss_function(val_predicted_target, val_sample_target)
            val_score = self.metrics(val_predicted_target, val_sample_target)
            total_val_loss += val_loss.item()
            total_val_score += val_score.item()
            del val_sample_img, val_sample_target, val_predicted_target, val_loss, val_score
            gc.collect()
        print('total_valid_score: ' + str(total_val_score / len(self.valid_loader)))
        torch.cuda.empty_cache()
        self.valid_losses.append(total_val_loss / len(self.valid_loader))
        self.valid_scores.append(total_val_score / len(self.valid_loader))
        return total_val_score / len(self.valid_loader)

    def predict(self):
        self.model.eval()
        total_test_loss = 0
        total_test_score = 0
        for index, test_batch in tqdm(
            enumerate(
                self.test_loader), total=len(
                self.test_loader)):
            test_sample_img, test_sample_target = test_batch
            test_sample_img = test_sample_img.to(self.device)
            test_sample_target = test_sample_target.to(self.device)
            del test_batch
            gc.collect()
            with torch.no_grad():
                test_predicted_target = self.model(test_sample_img)
            test_loss = self.loss_function(
                test_predicted_target, test_sample_target)
            test_score = self.metrics(test_predicted_target, test_sample_target)
            total_test_loss += test_loss.item()
            total_test_score += test_score.item()
            del test_sample_img, test_sample_target, test_predicted_target, test_loss, test_score
            gc.collect()
        print('test_score: ' + str(total_test_score / len(self.test_loader)))
        torch.cuda.empty_cache()
        self.test_score = total_test_score / len(self.test_loader)
        return total_test_score / len(self.test_loader)

    def save_checkpoint(self, name, epoch, val_score):
        if not os.path.exists('./results'):
            os.mkdir('./results')
        if not os.path.exists('./results/' + name):
            os.mkdir('./results/' + name)
        dill.dump(
            self,
            open(
                './results/' +
                name +
                '/epoch_' +
                str(epoch) +
                '_val_score=' +
                str(val_score) +
                '.pkl',
                'wb'))

    @staticmethod
    def load_best_checkpoint(name):
        checkpoints = sorted([checkpoint for checkpoint in os.listdir(
            './results/' + name) if checkpoint.startswith('epoch')])
        best_epoch = np.argmax([float(checkpoint.split('=')[1].split('.')[
                               1][:10]) for checkpoint in checkpoints])
        best_epoch = int(checkpoints[best_epoch].split('_')[1])
        print('best_epoch: ', best_epoch)
        best_checkpoint = [
            checkpoint for checkpoint in checkpoints if checkpoint.startswith(
                'epoch_' + str(best_epoch))][0]
        return dill.load(
            open(
                './results/' +
                name +
                '/' +
                best_checkpoint,
                'rb'))