import model
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import numpy as np
import time
import csv
import random
import os
import psutil

PREPARE_DATA = True  # Prepare data (only once)
SPLITTED = True # Split data into 200 samples each to avoid memory issues

if SPLITTED:
    maxIdxs = range(200, 2000+1, 200) # Indices
else:
    maxIdxs = ['']
snr_set = range(10,25+1)  # max is 10,25+1
sir_set = range(-5,15+1)  # max is -5,15+1

script_dir = os.path.dirname(__file__)
labels_zigbee = {0: 0, 11: 1, 12: 2, 13: 3, 14: 4}
labels_ble = {-1: 0, 0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13}
zigbee_channels = [0, 11, 12, 13, 14]
ble_channels = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
channel_models = ['B', 'C']
data_path = '../data/'

if PREPARE_DATA:
    for maxIdx in maxIdxs:
        for ch_model in channel_models:
            if SPLITTED:
              minIdx = maxIdx - 200
            
            relative_path = data_path+"/zigbee/"
            file_path = os.path.join(script_dir, relative_path)
            csi = []
            labels = []
            for c in zigbee_channels:
                if c == 0:
                    sir_set_actual = ['Inf']
                else:
                    sir_set_actual = sir_set

                for snr in snr_set:
                    for sir in sir_set_actual:
                        filename = str(c)+'_snr'+str(snr)+'_sir'+str(sir)+'_'+ch_model
                        csvfile = open(file_path+filename+'.csv', 'r', newline='')
                        csvreader = csv.reader(csvfile, delimiter=' ')
                        mem_usage = psutil.virtual_memory()[2]
                        if mem_usage > 95:
                            print('Memory usage too high:', mem_usage, 'during ZigBee at SNR', snr, 'and SIR', sir)
                            exit(1)
                        for i, line in enumerate(csvreader):
                            if SPLITTED and c != 0 and i < minIdx:
                                continue
                            if SPLITTED and c != 0 and i > maxIdx:
                                break
                            line = line[0].split(',')
                            if i % 2 == 0:
                              re = [int(float(l)) for l in line]
                            elif i % 2 == 1:
                              im = [int(float(l)) for l in line]
                              csi.append([re, im])
                              labels.append(labels_zigbee[c])

            relative_path = data_path+"BLE/"
            file_path = os.path.join(script_dir, relative_path)
            for c in ble_channels:
                if c == -1:
                    sir_set_actual = ['Inf']
                else:
                    sir_set_actual = sir_set

                for snr in snr_set:
                    for sir in sir_set_actual:
                        filename = str(c)+'_snr'+str(snr)+'_sir'+str(sir)+'_'+ch_model
                        csvfile = open(file_path+filename+'.csv', 'r', newline='')
                        csvreader = csv.reader(csvfile, delimiter=' ')
                        mem_usage = psutil.virtual_memory()[2]
                        if mem_usage > 95:
                            print('Memory usage too high:', mem_usage, 'during BLE at SNR', snr, 'and SIR', sir)
                            exit(1)
                        for i, line in enumerate(csvreader):
                            if SPLITTED and c != 0 and i < minIdx:
                                continue
                            if SPLITTED and c != 0 and i > maxIdx:
                                break
                            line = line[0].split(',')
                            if i % 2 == 0:
                              re = [int(float(l)) for l in line]
                            elif i % 2 == 1:
                              im = [int(float(l)) for l in line]
                              csi.append([re, im])
                              labels.append(labels_ble[c])


        # Randomize data
        data = list(zip(csi, labels))
        random.shuffle(data)
        csi, labels = zip(*data)
        csi = np.array(csi, dtype=np.float32)
        labels = np.array(labels)

        # Spliting data into training and testing sets
        train_csi = np.array(csi[:int(0.8*len(csi))])
        train_labels = np.array(labels[:int(0.8*len(labels))])
        test_csi = np.array(csi[int(0.8*len(csi)):])
        test_labels = np.array(labels[int(0.8*len(labels)):])

        relative_path = '../data/'
        file_path = os.path.join(script_dir, relative_path)
        post_fix = str(maxIdx)+'_snr'+str(min(snr_set))+'_'+str(max(snr_set))+'_sir'+str(min(sir_set))+'_'+str(max(sir_set)) 
        np.save(file_path+'train_multi_art_csi'+post_fix+'.npy', train_csi)
        np.save(file_path+'train_multi_art_labels'+post_fix+'.npy', train_labels)
        np.save(file_path+'test_multi_art_csi'+post_fix+'.npy', test_csi)
        np.save(file_path+'test_multi_art_labels'+post_fix+'.npy', test_labels)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSImodel = model.CNN()
# Define optimizer
opt = Adam(CSImodel.parameters(), lr=0.001)
lossFn = nn.NLLLoss()
CSImodel.train()

relative_path = '../data/'
file_path = os.path.join(script_dir, relative_path)
train_csi = np.load(file_path+'train_multi_csi_tester.npy')
train_labels = np.load(file_path+'train_multi_labels_tester.npy')
for maxIdx in maxIdxs:
    post_fix = str(maxIdx)+'_snr'+str(min(snr_set))+'_'+str(max(snr_set))+'_sir'+str(min(sir_set))+'_'+str(max(sir_set))
    train_csi_partial = np.load(file_path+'train_multi_art_csi'+post_fix+'.npy')
    train_labels_partial = np.load(file_path+'train_multi_art_labels'+post_fix+'.npy')
    print('Loaded', maxIdx)
    train_csi = np.concatenate((train_csi, train_csi_partial), axis=0)
    train_labels = np.concatenate((train_labels, train_labels_partial))

train_csi = torch.from_numpy(train_csi).type(torch.float32)
train_labels = torch.from_numpy(train_labels).type(torch.int64)

batch_size = 256
train_dataset = TensorDataset(train_csi, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 200
start_time = time.time()
try:
  for epoch in range(num_epochs):   
    totalTrainLoss = 0
    trainCorrect = 0

    for batch, labels_batch in train_loader:
      batch, labels_batch = (batch.to(device), labels_batch.to(device))

      # Forward pass
      out = CSImodel.forward(batch)
      # Calculate the loss
      loss = lossFn(out, labels_batch)
      
      # Zero out the gradients, perform the backpropagation, and update the weights
      opt.zero_grad()
      loss.backward()
      opt.step()

      totalTrainLoss += loss
      trainCorrect += (out.argmax(1) == labels_batch).sum().item()

    if epoch % 10 == 0:
        print(f"Time Taken: {time.time()-start_time:.1f}s, Epoch [{epoch+1}/{num_epochs}], Loss: {totalTrainLoss:.3f}, Accuracy: {trainCorrect/len(train_dataset):.3f}")  
        start_time = time.time()
except KeyboardInterrupt:
    pass

relative_path = 'CSImodel_snr'+str(min(snr_set))+'_'+str(max(snr_set))+'_sir'+str(min(sir_set))+'_'+str(max(sir_set))+'.pth'
file_path = os.path.join(script_dir, relative_path)
torch.save(CSImodel.state_dict(), file_path)

# Evaluation on the test set
relative_path = '../data/'
file_path = os.path.join(script_dir, relative_path)
test_csi = np.load(file_path+'test_multi_csi_tester.npy')
test_labels = np.load(file_path+'test_multi_labels_tester.npy')
for maxIdx in maxIdxs:
    post_fix = str(maxIdx)+'_snr'+str(min(snr_set))+'_'+str(max(snr_set))+'_sir'+str(min(sir_set))+'_'+str(max(sir_set))
    test_csi_partial = np.load(file_path+'test_multi_art_csi'+post_fix+'.npy')
    test_labels_partial = np.load(file_path+'test_multi_art_labels'+post_fix+'.npy')
    test_csi = np.concatenate((test_csi, test_csi_partial), axis=0)
    test_labels = np.concatenate((test_labels, test_labels_partial))

test_csi = torch.from_numpy(test_csi).type(torch.FloatTensor)
test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)
test_dataset = TensorDataset(test_csi, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valCorrect = 0
with torch.no_grad():
  st = time.perf_counter()
  for batch, labels_batch in test_loader:
      batch, labels_batch = (batch.to(device), labels_batch.to(device))
      out = CSImodel.forward(batch)
      valCorrect += (out.argmax(1) == labels_batch).type(torch.float).sum().item()

print(f"Test Accuracy: {valCorrect/len(test_dataset):.3f}")
print(f"Time Taken To Test: {time.perf_counter() - st}")
    
