import math
import torch
from torch import nn
import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import get_linear_schedule_with_warmup
from model.baseline import Encoder, Classifier
from utils.read_data import statistic
from utils.replay_dataloader import MemoryLoader
from utils.mtl import *
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import random


class Trainer:

    def __init__(self, args):
        self.args = args
        self.task_num = 0
        self.past_classifier = None
        self.encoder = Encoder(args)
        self.classifier = Classifier(args)
        self.buffer_distribution = {}
        self.key_mixture = {}
        self.buffer_embedding = {}
        self.past_memory = None
        self.past_label_set = None

    def new_task(self, train_dataset, test_dataset, num_labels):
        self.task_num += 1
        self.past_memory = self.buffer_embedding
        self.curr_label_set = set(train_dataset.labels)
        self.past_label_set = set(self.buffer_embedding.keys())
        for i in self.curr_label_set:
            self.buffer_distribution[i] = [] 
            self.key_mixture[i] = None
            self.buffer_embedding[i] = []
        # expand classifier and prefix
        self.classifier.new_task(num_labels)
        # train classifier with new dataset
        self.training(train_dataset)
        # evaluating
        self.evaluating(test_dataset)

    def training(self, dataset, test_dataset=None):
        
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.args.lr_list[self.task_num - 1], weight_decay=0.0)
        num_training_steps = len(
            loader) * self.args.epochs_list[self.task_num - 1]
        num_warmup_steps = num_training_steps * self.args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps)
        self.encoder.cuda()
        self.classifier.cuda()

        cur_embeding = []
        cur_label = []
        for idx, batch in enumerate(tqdm(loader, desc=f"Forward current data")):
            
            # print("Forward current data...")
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                get_prelogits=True
            )
            cur_embeding.append(outputs)
            cur_label.append(labels)
            labels = labels.cpu()
            for i, _ in enumerate(labels):
                self.buffer_distribution[labels[i].item()].append(outputs[i].cpu())
        
        for label in self.curr_label_set:
            self.key_mixture[label] = GaussianMixture(n_components=1, random_state=42).fit(self.buffer_distribution[label])
            # if self.args.gmm_num_components == 1:
            self.key_mixture[label].weights_[0] = 1.0
        #Sample prelogits for each label
        for i, label in enumerate(self.curr_label_set):
            replay_embedding =  self.key_mixture[label].sample(512)[0].astype("float32")
            self.buffer_embedding[label].append(torch.tensor(replay_embedding))
        
        if self.task_num ==1:
            self.classifier.train()
            for epoch in range(self.args.epochs_list[self.task_num - 1]):
                
                correct, total = 0, 0
                total_loss = 0
                for idx, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
                    #Finetune classifier on current data
                    # _, _, labels = batch
                    labels = cur_label[idx]
                    labels = labels.cuda()
                    optimizer.zero_grad()
                    logits = self.classifier(cur_embeding[idx].cuda())
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, logits.shape[-1]), labels.view(-1))
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pred = torch.argmax(logits, dim=1)
                    correct += torch.sum(pred == labels).item()
                    total += len(labels)

                print(f"Epoch {epoch} Training Accuracy: {correct/total}")
                print(f"Epoch {epoch} Average Loss: {total_loss/len(loader)}")
        else:
            # self.past_classifier = self.classifier.get_cur_classifer()
            self.past_classifier = deepcopy(self.classifier)
            self.past_classifier.cuda()
            self.classifier.train()
            for epoch in range(10):
                #Finetune classifier on current GMM data
                correct, total = 0, 0
                total_loss = 0
                for idx, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
                    #Finetune classifier on current data
                    # _, _, labels = batch
                    labels = cur_label[idx]
                    labels = labels.cuda()
                    optimizer.zero_grad()
                    logits = self.classifier(cur_embeding[idx].cuda())
                    # logits[:, :self.classifier.old_num_labels] = -1e4
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, logits.shape[-1]), labels.view(-1))
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pred = torch.argmax(logits, dim=1)
                    correct += torch.sum(pred == labels).item()
                    total += len(labels)

                print(f"Epoch {epoch} Training Accuracy: {correct/total}")
                # print(f"Epoch {epoch} Average Loss: {total_loss/len(loader)}")

            # self.finetuned_classifier = self.classifier.get_cur_classifer()
            self.finetuned_classifier = deepcopy(self.classifier)
            self.finetuned_classifier.cuda()
            # self.classifier = self.past_classifier.get_cur_classifer()
            self.classifier = deepcopy(self.past_classifier)
            self.classifier.cuda()
            optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.args.lr_list[self.task_num - 1], weight_decay=0.0)
            scheduler = get_linear_schedule_with_warmup(
                                    optimizer, num_warmup_steps, num_training_steps)
            replay_loader = MemoryLoader(self.past_memory, 32, self.past_label_set)

            self.classifier.train()
            self.finetuned_classifier.eval()
            self.past_classifier.eval()
            cur_loader = MemoryLoader(self.buffer_embedding, 32, self.curr_label_set)
            for epoch in range(self.args.epochs_list[self.task_num - 1]):

                correct, total = 0, 0
                total_loss = 0
                total_distill_loss = 0
                total_distill_loss_mem = 0
                total_loss_mem = 0
                # for idx, (batch, replay_batch) in enumerate(tqdm(zip(cur_loader,replay_loader), desc=f"Training Epoch {epoch}")):
                for idx, batch in enumerate(tqdm(cur_loader, desc=f"Training Epoch {epoch}")):
                    cur_embed, cur_labels = batch
                    #Distill current classifier vs finetuned classifier
                    optimizer.zero_grad()
                    cur_labels = torch.tensor(cur_labels).cuda()
                    cur_reps = self.classifier(cur_embed.cuda())
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        cur_reps.view(-1, cur_reps.shape[-1]), cur_labels.view(-1))

                    with torch.no_grad():
                        past_reps = self.finetuned_classifier(cur_embed.cuda())
                    distill_loss = self.distill_loss(
                        cur_reps, past_reps) 
                    #Forwar Memory
                    replay_embed, replay_labels = next(iter(replay_loader))
                    # replay_embed, replay_labels = replay_batch
                    replay_labels = torch.tensor(replay_labels).cuda()
                    replay_reps = self.classifier(replay_embed.cuda())
                    # replay_reps[:,self.classifier.old_num_labels:] = -1e4
                    loss_mem = loss_fct(
                        replay_reps.view(-1, replay_reps.shape[-1]), replay_labels.view(-1))
                    
                    with torch.no_grad():
                        past_replay_reps = self.past_classifier(replay_embed.cuda())
                    # past_replay_reps[:,self.classifier.old_num_labels:] = -1e4
                    distill_loss_mem = self.distill_loss(
                        replay_reps, past_replay_reps)
                    
                    # Backward and optimize
                    loss.backward(retain_graph=True)
                    loss_shared_grad = []
                    for name, param in self.classifier.named_parameters():
                        if param.grad is None:
                            continue
                        else:
                            loss_shared_grad.append(param.grad.detach().data.clone().flatten())
                        param.grad.zero_()
                    loss_shared_grad = torch.cat(loss_shared_grad, dim=0)

                    distill_loss.backward()
                    distill_shared_grad = []
                    for name, param in self.classifier.named_parameters():
                        if param.grad is None:
                            continue
                        else:
                            distill_shared_grad.append(param.grad.detach().data.clone().flatten())
                        param.grad.zero_()
                    distill_shared_grad = torch.cat(distill_shared_grad, dim=0)

                    loss_mem.backward(retain_graph=True)

                    loss_mem_shared_grad = []
                    for name, param in self.classifier.named_parameters():
                        if param.grad is None:
                            continue
                        else:
                            loss_mem_shared_grad.append(param.grad.detach().data.clone().flatten())
                        param.grad.zero_()
                    loss_mem_shared_grad = torch.cat(loss_mem_shared_grad, dim=0)
                    distill_loss_mem.backward()
                    distill_mem_shared_grad = []
                    for name, param in self.classifier.named_parameters():
                        if param.grad is None:
                            continue
                        else:
                            param.grad = param.grad
                            distill_mem_shared_grad.append(param.grad.detach().data.clone().flatten())
                        param.grad.zero_()
                    distill_mem_shared_grad = torch.cat(distill_mem_shared_grad, dim=0)



                    # mtl_output = AUGD(torch.stack([distill_shared_grad, loss_shared_grad, loss_mem_shared_grad, distill_mem_shared_grad]))
                    # mtl_output = AUGD(torch.stack([distill_shared_grad, distill_mem_shared_grad]))
                    mtl_output = CAGrad(torch.stack([distill_shared_grad, loss_shared_grad, loss_mem_shared_grad, distill_mem_shared_grad]))

                    # mtl_output = AUGD(torch.stack([loss_shared_grad, loss_mem_shared_grad, distill_shared_grad]))
                    # mtl_output = AUGD(torch.stack([loss_shared_grad, loss_mem_shared_grad]))
                    # mtl_output = CAGrad(torch.stack([loss_shared_grad, loss_mem_shared_grad]))
                    shared_grad = mtl_output["updating_grad"]
                    # print("Alpha: ", mtl_output["alpha"])
                    # print("Norm_grad", mtl_output["norm_grads"])

                    total_length = 0
                    for name, param in self.classifier.named_parameters():
                        # if 'cur' not in name:
                        length = param.numel()
                        param.grad.data = shared_grad[
                            total_length : total_length + length
                        ].reshape(param.shape)
                        total_length += length

                    # training_loss = 5*loss + 5*distill_loss + loss_mem + 10*distill_loss_mem
                    # training_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pred = torch.argmax(cur_reps, dim=1)
                    correct += torch.sum(pred == cur_labels).item()
                    total += len(cur_labels)
                    total_loss += loss.item()
                    total_distill_loss += distill_loss.item()
                    total_distill_loss_mem += distill_loss_mem.item()
                    total_loss_mem += loss_mem.item()
                    

                print(f"Epoch {epoch} Training Accuracy: {correct/total}")
                print(f"Epoch {epoch} Average Loss: {total_loss/len(loader)}")
                print(f"Epoch {epoch} Average mem_Loss: {total_loss_mem/len(loader)}")
                print(f"Epoch {epoch} Average total_distill_loss_mem: {total_distill_loss_mem/len(loader)}")
                print(f"Epoch {epoch} Average total_distill_loss: {total_distill_loss/len(loader)}")




           


            # print(f"Epoch {epoch} Training Accuracy: {correct/total}")
            # print(f"Epoch {epoch} Average Loss: {total_loss/len(loader)}")
            # if test_dataset is not None:
            #     self.evaluating(test_dataset)

    def evaluating(self, dataset):
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        self.classifier.cuda()
        self.encoder.cuda()
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for _, batch in enumerate(tqdm(loader, desc="Evaluating")):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                outputs = self.classifier(self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ))
                logits = outputs
                pred = torch.argmax(logits, dim=1)
                correct += torch.sum(pred == labels).item()
                total += len(labels)

            print(f"Evaluating Accuracy: {correct/total: .4f}")
        return correct / total

    def evaluating_for_datsets(self, datasets):
        eval_acc = [self.evaluating(dataset)
                    for _, dataset in enumerate(datasets)]
        print(f"-" * 50)
        print(f"Average Evaluating Accuracy: {np.mean(eval_acc): .4f}")
        print(f"{np.around(eval_acc, 4)}")
        print(f"-" * 50)
        return eval_acc
    

    def distill_loss(self, pred, soft, T=2):
        """
        args:
            pred: logits of student model, [n, old_num_labels]
            soft: logits of teacher model, [n, old_num_labels]
            T: temperature
        return:
            loss: distillation loss (batch mean)
        """
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
import random

def sample_batch(memory, batch_size, labels):

  labels = list(labels)
  inputs_batch = []
  labels_batch = []

  while len(inputs_batch) < batch_size:
    label = random.choice(labels)

    input_ids = random.choice(memory[label][0])

    inputs_batch.append(input_ids)
    labels_batch.append(label)

  return inputs_batch, labels_batch

