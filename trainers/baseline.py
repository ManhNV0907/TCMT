# learning without forgetting
import math
import torch
from torch import nn
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import get_linear_schedule_with_warmup
from model.baseline import Encoder, Classifier
from utils.read_data import statistic
from sklearn.mixture import GaussianMixture

class Trainer:

    def __init__(self, args):
        self.args = args
        # if args.method == "baseline":
            # self.model = Encoder(args)
        self.task_num = 0
        # past classifier
        self.past_classifier = None
        # Classifier
        self.encoder = Encoder(args)
        self.classifier = Classifier(args)
        self.buffer_distribution = {}
        self.key_mixture = {}
        self.buffer_embedding = {}

    def new_task(self, train_dataset, test_dataset, num_labels):
        self.task_num += 1
        self.curr_label_set = set(train_dataset.labels)
        for i in self.curr_label_set:
            self.buffer_distribution[i] = [] 
            self.key_mixture[i] = None
            self.buffer_embedding[i] = []
        # expand classifier and prefix
        self.classifier.new_task(num_labels)
        # train classifier with new dataset
        self.training(train_dataset)
        #train classifier with GMM data set
        #distillation
        # self.training(train_dataset, test_dataset)
        # evaluating
        # self.evaluating(test_dataset)

    def training(self, dataset, test_dataset=None):
        
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr_list[self.task_num - 1], weight_decay=0.0)
        num_training_steps = len(
            loader) * self.args.epochs_list[self.task_num - 1]
        num_warmup_steps = num_training_steps * self.args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps)
        # self.model.cuda()
        self.encoder.cuda()
        self.classifier.cuda()

        # X_embedding = {}
        new_data = copy.deepcopy(dataset)
        cur_embeding = []
        for idx, batch in enumerate(tqdm(loader, desc=f"Forward current data")):
            
            print("Forward current data...")
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
            labels = labels.cpu()
            for i, _ in enumerate(labels):
                self.buffer_distribution[labels[i].item()].append(outputs[i].cpu())
        
        for label in self.curr_label_set:
            self.buffer_distribution[label] =  torch.cat(self.buffer_distribution[label], dim=0).reshape(-1, 1)
        
            self.key_mixture[label] = GaussianMixture(n_components=1, random_state=42).fit(self.buffer_distribution[label].cpu().detach().numpy())
            # if self.args.gmm_num_components == 1:
            self.key_mixture[label].weights_[0] = 1.0
                # print()
        #Sample prelogits 
        for i, label in enumerate(self.curr_label_set):
            replay_embedding =  self.key_mixture[label].sample(100 * 256)[0].astype("float32")
            self.buffer_embedding[label].append(replay_embedding)
        cur_embeding = torch.cat(cur_embeding, dim=0).cpu()
        for i in range(len(new_data)):
            new_data[idx]["embedding"] = cur_embeding[i]
        if self.task_num > 1:
            self.past_classifier = copy.deepcopy(self.classifier())
        for epoch in range(self.args.epochs_list[self.task_num - 1]):
            self.classifier.train()
            correct, total = 0, 0
            total_loss = 0
            for idx, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
                #Finetune classifier on current data
                # _, _, labels = batch
                # labels = labels.cuda()
                optimizer.zero_grad()
                logits = self.classifier(new_data[idx]["embedding"].cuda())
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, logits.shape[-1]), labels.view(-1))
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                _, _, labels = batch
                labels = labels.cuda()
                pred = torch.argmax(outputs.logits, dim=1)
                correct += torch.sum(pred == labels).item()
                total += len(labels)


           


            print(f"Epoch {epoch} Training Accuracy: {correct/total}")
            print(f"Epoch {epoch} Average Loss: {total_loss/len(loader)}")
            if test_dataset is not None:
                self.evaluating(test_dataset)

    # def optimization(self, batch, optimizer, scheduler):
    #     input_ids, attention_mask, labels = batch
    #     input_ids = input_ids.cuda()
    #     attention_mask = attention_mask.cuda()
    #     labels = labels.cuda()
    #     optimizer.zero_grad()
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels
    #     )
    #     loss = outputs.loss
    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()
    #     return outputs

    def evaluating(self, dataset):
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for _, batch in enumerate(tqdm(loader, desc="Evaluating")):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
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
    


   