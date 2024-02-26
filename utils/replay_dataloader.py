import random
import torch
from torch.utils.data import Dataset, DataLoader

class MemoryDataset(Dataset):
    def __init__(self, memory, labels):
        self.memory = memory
        self.labels = list(labels)
        self.inputs = []
        self.labels_list = []

        # for label in self.labels:
        #     input_ids_list = self.memory.get(label, [])[0]
        #     print(len(input_ids_list))
        #     print(len(input_ids_list[0]))
        #     self.inputs.extend(input_ids_list)
        #     self.labels_list.extend([label] * len(input_ids_list))
        for label in self.labels:
            input_ids_list = self.memory.get(label, [])
            for input_ids in input_ids_list:
                self.inputs.extend(input_ids)
                self.labels_list.extend(label)
                # self.labels_list.extend([label] * len(input_ids_list))



    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = self.inputs[index]
        label = self.labels_list[index]
        return input_ids, label

def MemoryLoader(memory, batch_size, labels):
    dataset = MemoryDataset(memory, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
    # inputs_batch, labels_batch = next(iter(dataloader))
    # return inputs_batch, labels_batch