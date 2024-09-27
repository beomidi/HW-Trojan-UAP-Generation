#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils.utils import create_directory
from utils.utils import get_splited_list_of_files_and_scaler_HT
from utils.utils import Data_Generator, summrize
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import sys, os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import tensorflow as tf
import keras
from random import shuffle
import utils.utils_HT as ht
from utils.utils import get_splited_list_of_files_and_scaler_HT
from utils.utils import Data_Generator, summrize
from keras_contrib.layers import InstanceNormalization
from sklearn.preprocessing import normalize
import numpy as np
import os
from scipy.signal import butter, lfilter, firwin, remez, kaiser_atten, kaiser_beta
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
import statistics
import math
from random import randrange
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import copy
import pdb
from tqdm import tqdm
from scipy.signal import firwin
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore")
import argparse
import zipfile
import gdown

def lowpass_firwin(data , ntaps, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    b = firwin(ntaps, highcut, nyq=nyq, pass_zero=True,
                  window=window, scale=False)
    
    y = list(np.convolve(np.ravel(np.array(data)), np.ravel(b), mode='same'))
    return y


def Preprocessing(benchmark = 'AES-T700', model_name = 'HTnet'):
    match model_name:
        case 'HTnet':
            url_regular = 'https://drive.google.com/uc?export=download&id=19c7g-MPtixhJfC3fohjPtHlfq1LepvD9'
            url_at = 'https://drive.google.com/uc?export=download&id=1BDKHnF3xKOAZsKxqE82c7_UOrY1kzZuS'
        case 'ResNet-18':
            url_regular = 'https://drive.google.com/uc?export=download&id=1Jod-TXjtz_xQxSUzeJu4G9hQoNCv1jQ6'
            url_at = 'https://drive.google.com/uc?export=download&id=1gRAwVs5G8XjTsnvHsUgNngRZ-qdrJP5X'
        case 'VGG-11':
            url_regular = 'https://drive.google.com/uc?export=download&id=1S7ePyec-ol7C_JF4VWIaIkryvV6etiTW'
            url_at = 'https://drive.google.com/uc?export=download&id=1qbl7UYnfHH65Nie9sSLfCAVex1ZDTwrp'
        case 'SVM':
            url_regular = 'https://drive.google.com/uc?export=download&id=1syjXUX5_v4pVnp0VG4D_JVg9PF9WBrHS'
            url_at = 'https://drive.google.com/uc?export=download&id=168gD8k0pMX5ffFdZ8WT2uQMpIf04rE-x'
        case default:
            sys.exit('Error: ' + model_name + ' is not supported.')


    if not os.path.isdir('./trained_models/regular_models/pytorch/' + model_name):
        gdown.download(url_regular, './trained_models/regular_models/pytorch/', quiet=False)
        with zipfile.ZipFile('./trained_models/regular_models/pytorch/' + model_name + '.zip', 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                try:
                    zip_ref.extract(member, './trained_models/regular_models/pytorch/')
                except zipfile.error as e:
                    pass
                
    if not os.path.isdir('./trained_models/at_models/pytorch/' + model_name):
        gdown.download(url_at, './trained_models/at_models/pytorch/', quiet=False)
        with zipfile.ZipFile('./trained_models/at_models/pytorch/' + model_name + '.zip', 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                try:
                    zip_ref.extract(member, './trained_models/at_models/pytorch/')
                except zipfile.error as e:
                    pass
                
    match benchmark:
        case 'AES-T400':
            url = 'https://drive.google.com/uc?export=download&id=1sxfSfYc-T_XCJCHENxbpqlvgDXvL7Ma_'
        case 'AES-T500':
            url = 'https://drive.google.com/uc?export=download&id=153kahK2z7O16rVwtyt0pG3PXnmgf9gJD'
        case 'AES-T600':
            url = 'https://drive.google.com/uc?export=download&id=1d_aixtyDY1qC-8Ij3XlV7HYyikj4MKCp'
        case 'AES-T700':
            url = 'https://drive.google.com/uc?export=download&id=1AR3WDI0c6iwlpyOspeGKlOudHcutdBPD'
        case 'AES-T800':
            url = 'https://drive.google.com/uc?export=download&id=1ZlFeANl4zllhKfjGqcUsdatgq3cRuta_'
        case 'AES-T1800':
            url = 'https://drive.google.com/uc?export=download&id=1HVBMFxq-XagymyPfK2n-XdAWrdZYFHF8'
        case default:
            sys.exit('Error: ' + benchmark + ' is not supported.')
                
    if not os.path.isdir('./dataset/' + benchmark + '_power_Temp25C'):
        gdown.download(url, './dataset/', quiet=False)
        with zipfile.ZipFile('./dataset/' + benchmark + '_power_Temp25C.zip', 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                try:
                    zip_ref.extract(member, './dataset/')
                except zipfile.error as e:
                    pass
    
    
def KerasDataPrep(benchmark = 'AES-T700', number_of_samples = 40000, batch_size = 20):

    name_bms = benchmark + '_power_Temp25C'
    dir2bms_folder = './dataset/'

    dirs_to_files_train, dirs_to_files_test, label_train, label_test, scaler, input_shape =                 get_splited_list_of_files_and_scaler_HT(dir2bms_folder = dir2bms_folder, name_bms=[name_bms], 
                                                        use_enabled_trojan_folder = False,folder_numbers= [1,2], number_of_training_for_scaler=100, number_of_samples = number_of_samples)
    
    train_generator = Data_Generator(dirs_to_files_train, label_train, batch_size=batch_size,
                                         dir2bms_folder=dir2bms_folder, scaler=scaler)
    val_generator = Data_Generator(dirs_to_files_test, label_test, batch_size=batch_size,
                                           dir2bms_folder=dir2bms_folder, scaler=scaler)
    nb_classes = len(np.unique(np.concatenate((label_train, label_test), axis=0)))

    data_train = (np.array([
                    np.loadtxt(dir_to_file, delimiter='\0')
                      for dir_to_file in train_generator.dirs_to_files]))

    data_test = (np.array([
                    np.loadtxt(dir_to_file, delimiter='\0')
                      for dir_to_file in val_generator.dirs_to_files]))
    

    data_train = tf.cast(tf.reshape(tf.convert_to_tensor([data_train]), np.shape(data_train)), tf.float32)
    label_train = tf.cast(tf.reshape(tf.convert_to_tensor([label_train]), np.shape(label_train)), tf.float32)
    data_test = tf.cast(tf.reshape(tf.convert_to_tensor([data_test]), np.shape(data_test)), tf.float32)
    label_test = tf.cast(tf.reshape(tf.convert_to_tensor([label_test]), np.shape(label_test)), tf.float32)
    
    return data_train, label_train, data_test, label_test, input_shape, nb_classes
            
def TorchLoadModel(model_name, benchmark):

    torch_model = torch.load('trained_models/regular_models/pytorch/' + model_name + '/' + benchmark + '/' + benchmark + '.pt', map_location=torch.device('cpu'));
    torch_model.to(device)
    return torch_model
    
def TorchDataPrep(model_name, input_data, label):

    if(model_name in ['HTnet', 'SVM']):
        batch_size = 20
        my_dataset = TensorDataset(torch.FloatTensor(input_data.numpy()), torch.FloatTensor(label.numpy())) 
        data_loader = DataLoader(my_dataset, batch_size=batch_size)
    else:
        batch_size = 1
        my_dataset  = TensorDataset((torch.reshape(torch.FloatTensor(input_data.numpy()), [input_data.shape[0],1,50, 50]).repeat([1,3,1,1])), torch.FloatTensor(label.numpy())) 
        data_loader = DataLoader(my_dataset, batch_size=batch_size)
    return data_loader

def ModelEvaluation(model_name, torch_model, data_loader):
    classes = (0,1)
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct_pred_model = 0
    total_pred_model   = 0
    pred = []
    target = []

    for _, (data_test_batched, label_test_batched) in enumerate(data_loader):

        if(model_name == 'SVM'):
            pred = torch.argmax(torch_model.predict_proba(data_test_batched.type(torch.DoubleTensor).to(device)), axis = 1)
            target  = torch.argmax(label_test_batched, axis = 1) 
        else:
            pred = torch.argmax(torch_model(data_test_batched.to(device)), axis = 1)
            target  = torch.argmax(label_test_batched, axis = 1) 

        for label, prediction in zip(target, pred.cpu().detach().numpy()):
            if label == prediction:
                correct_pred[classes[label]] += 1
                correct_pred_model += 1
            total_pred[classes[label]] += 1
            total_pred_model += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5d} is {accuracy:.1f} %')

    accuracy_model = 100 * float(correct_pred_model) / total_pred_model
    print(f'Model Accuracy:          {accuracy_model:.1f}%')
    
        
def SyncAdversarialPatchGen(model_name, torch_model , data_loader, nb_epoch=5, eps=2, resolution = 0.1, gn=False, filter_in_loop=False, fs=100, fh=20, at_mode = False):
    _, (data_test_batched, _)  = next(enumerate(data_loader))
    batch_delta = torch.zeros_like(data_test_batched).to(device)

    delta = []
    if(model_name in ['SVM']): 
        delta = batch_delta[0]
    else:
        delta = batch_delta[0][0]

    losses = []
    batch_delta.requires_grad_()

    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    def clamped_loss(output, target):
              loss = torch.mean(loss_fn(output, target))
              return loss

    
    for epoch in tqdm(range(nb_epoch), disable=at_mode):
#         print('epoch %i/%i' % (epoch + 1, nb_epoch))
        eps_step = resolution
        for _, (data_test_batched, label_test_batched) in enumerate(data_loader):
            y_target = torch.Tensor(np.repeat([[1.0, 0.0]], label_test_batched.shape[0], axis=0)).to(device)
          
            if batch_delta.grad is not None:
                batch_delta.grad.data.zero_()
                if(model_name in ['HTnet', 'SVM']): 
                    batch_delta.data = delta.unsqueeze(0).repeat([data_test_batched.shape[0], 1])
                else:
                    batch_delta.data = torch.reshape(delta.unsqueeze(0), [1,1, 50, 50]).repeat([1, 3, 1 , 1])
              
            trace = torch.stack([data_test_batched[i].to(device) + batch_delta[i] if torch.argmax(label_test_batched, axis = 1)[i] == 1
              else data_test_batched[i].to(device) for i in range(data_test_batched.shape[0])])
          
            if gn:
                trace.data = trace + torch.randn(trace.shape).to(device)
          
            if(model_name in ['HTnet', 'SVM']):  
                 if filter_in_loop:
                    trace_tmp = [lowpass_firwin(x, 1024 , fh, fs, window='hamming') for x in (trace.cpu().detach().numpy())]
                    trace.data = torch.Tensor(trace_tmp).to(device)
            else:
                 if filter_in_loop:
                    trace_tmp = torch.reshape(trace[0][0], [1, 2500])
                    trace_tmp = [lowpass_firwin(x, 1024 , fh, fs, window='hamming') for x in (trace_tmp.cpu().detach().numpy())]
                    trace.data= torch.reshape(torch.Tensor(trace_tmp).to(device), [1, 1, 50, 50]).repeat([data_test_batched.shape[0],3,1,1]).to(device)

                      
            outputs = []
            if(model_name in ['SVM']):      
                outputs = torch_model.predict_proba(trace.type(torch.DoubleTensor))
            else:
                outputs = torch_model(trace)

            loss = -clamped_loss(outputs, y_target)
            losses.append(torch.mean(loss.detach().cpu()))
            loss.backward()

            if(batch_delta.grad is not None):
                grad_sign = []
                if(model_name in ['HTnet', 'SVM']):
                    grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
                else:
                    grad_sign = batch_delta.grad.data.mean(dim = 0).mean(dim = 0).sign()
                    grad_sign = torch.reshape(grad_sign, [1, 2500])
                    delta = torch.reshape(delta, [1, 2500])
                    
                delta = delta + grad_sign * eps_step 
                if filter_in_loop:
                    delta.data = torch.Tensor(lowpass_firwin(delta.cpu().detach().numpy(), 1024 , fh, fs, window='hamming')).to(device)
                delta = torch.clamp(delta, 0, eps)
                batch_delta.grad.data.zero_()
    return delta, losses


def SyncModelEvaluation(model_name, torch_model, data_loader, delta, gn = False, filter_in_loop= False, fs=100, fh=20):
    classes = (0,1)
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct_pred_model = 0
    total_pred_model   = 0
    pred = []
    target = []

    for _, (data_test_batched, label_test_batched) in enumerate(data_loader):
        
        if(model_name in ['HTnet', 'SVM']):
            if np.isnan(delta[0].cpu().detach().numpy()):
                delta = torch.zeros_like(delta).to(device)
        else:
            if np.isnan(delta[0].cpu().detach().numpy()):
                delta = torch.zeros_like(delta).to(device)
        
        noise = []
        if(model_name in ['HTnet', 'SVM']): 
            noise = torch.Tensor.repeat(torch.reshape(delta, [1, data_test_batched.shape[1]]), [data_test_batched.shape[0],1]).to(device)
        else:
            noise = torch.reshape(delta, [data_test_batched.shape[0], 1, 50, 50]).repeat([data_test_batched.shape[0],3,1,1]).to(device)


        trace = torch.stack([data_test_batched[i].to(device) + noise[i] if torch.argmax(label_test_batched, axis = 1)[i] == 1
              else data_test_batched[i].to(device) for i in range(data_test_batched.shape[0])])

        if gn:
            trace.data = trace + torch.randn(trace.shape).to(device)

        if(model_name in ['HTnet', 'SVM']):  
            if filter_in_loop:
                trace_tmp = [lowpass_firwin(x, 1024 , fh, fs, window='hamming') for x in (trace.cpu().detach().numpy())]
                trace.data = torch.Tensor(trace_tmp).to(device)
        else:
            if filter_in_loop:
                trace_tmp = torch.reshape(trace[0][0], [1, 2500])
                trace_tmp = [lowpass_firwin(x, 1024 , fh, fs, window='hamming') for x in (trace_tmp.cpu().detach().numpy())]
                trace.data= torch.reshape(torch.Tensor(trace_tmp).to(device), [1, 1, 50, 50]).repeat([data_test_batched.shape[0],3,1,1]).to(device)


        if(model_name == 'SVM'):
            pred = torch.argmax(torch_model.predict_proba(trace.type(torch.DoubleTensor).to(device)), axis = 1)
            target  = torch.argmax(label_test_batched, axis = 1) 
        else:
            pred = torch.argmax(torch_model(trace.to(device)), axis = 1)
            target  = torch.argmax(label_test_batched, axis = 1) 

        class_accuracy = []

        for label, prediction in zip(target, pred.cpu().detach().numpy()):
            if label == prediction:
                correct_pred[classes[label]] += 1
                correct_pred_model += 1
            total_pred[classes[label]] += 1
            total_pred_model += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5d} is {accuracy:.1f} %')
        class_accuracy.append(accuracy)

    accuracy_model = 100 * float(correct_pred_model) / total_pred_model
    print(f'Model Accuracy:          {accuracy_model:.1f}%')
    
    return class_accuracy
    

def UnsyncAdversarialPatchGen(model_name, torch_model , data_loader, nb_epoch=5, eps=6, resolution=0.1, gn=False, filter_in_loop=False, fs=100, fh=20):

    _, (data_test_batched, _)  = next(enumerate(data_loader))

    batch_delta = torch.zeros_like(data_test_batched).to(device)
    delta = []
    if(model_name in ['SVM']): 
        delta = batch_delta[0]
    else:
        delta = batch_delta[0][0]

    losses = []
    batch_delta.requires_grad_()

    loss_fn = nn.CrossEntropyLoss(reduction = 'none')

    def clamped_loss(output, target):
        loss = torch.mean(loss_fn(output, target))
        return loss

    
    for epoch in tqdm(range(nb_epoch)):
        eps_step = resolution

        for _, (data_test_batched, label_test_batched) in enumerate(data_loader):
            y_target = torch.Tensor(np.repeat([[1.0, 0.0]], label_test_batched.shape[0], axis=0)).to(device)
             
            shift = [randrange(10) for i in range(data_test_batched.shape[0])] 

            if batch_delta.grad is not None:
                batch_delta.grad.data.zero_()
                if(model_name in ['HTnet', 'SVM']): 
                    batch_delta.data = delta.unsqueeze(0).repeat([data_test_batched.shape[0], 1])
                else:
                    batch_delta.data = torch.reshape(delta.unsqueeze(0), [1,1, 50, 50]).repeat([1, 3, 1 , 1])

            if(model_name in ['HTnet', 'SVM']): 
                a_p = torch.stack([torch.concat([batch_delta[i][shift[i]:]
                                    ,batch_delta[i][:shift[i]]]) for i in range(data_test_batched.shape[0])])   
            else:
                batch_delta_tmp = torch.reshape(batch_delta, [1,3,1,2500])
                batch_delta_tmp = batch_delta_tmp[0][0]
                a_p = torch.stack([torch.concat([batch_delta_tmp[i][shift[i]:]
                                   ,batch_delta_tmp[i][:shift[i]]]) for i in range(data_test_batched.shape[0])])      
                a_p = torch.reshape(a_p, [1,1,50,50]).repeat([data_test_batched.shape[0], 3, 1 , 1]).to(device) 

            trace = torch.stack([data_test_batched[i].to(device) + a_p[i] if torch.argmax(label_test_batched, axis = 1)[i] == 1
                else data_test_batched[i].to(device) for i in range(data_test_batched.shape[0])])

            if gn:
                trace.data = trace + torch.randn(trace.shape).to(device)

            if(model_name in ['HTnet', 'SVM']):  
                 if filter_in_loop:
                    trace_tmp = [lowpass_firwin(x, 1024 , fh, fs, window='hamming') for x in (trace.cpu().detach().numpy())]
                    trace.data = torch.Tensor(trace_tmp).to(device)
            else:
                 if filter_in_loop:
                    trace_tmp = torch.reshape(trace[0][0], [1, 2500])
                    trace_tmp = [lowpass_firwin(x, 1024 , fh, fs, window='hamming') for x in (trace_tmp.cpu().detach().numpy())]
                    trace.data= torch.reshape(torch.Tensor(trace_tmp).to(device), [batch_size, 1, 50, 50]).repeat([data_test_batched.shape[0],3,1,1]).to(device)


            outputs = []
            if(model_name in ['SVM']):      
                outputs = torch_model.predict_proba(trace.type(torch.DoubleTensor))
            else:
                outputs = torch_model(trace)
            loss = -clamped_loss(outputs, y_target)
            losses.append(torch.mean(loss.detach().cpu()))
            loss.backward()

            if(batch_delta.grad is not None):
                grad_sign = []
                if(model_name in ['HTnet', 'SVM']):
                    grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
                else:
                    grad_sign = batch_delta.grad.data.mean(dim = 0).mean(dim = 0).sign()
                    grad_sign = torch.reshape(grad_sign, [1, 2500])
                    delta = torch.reshape(delta, [1, 2500])
                    
                delta = delta + grad_sign * eps_step 
                if filter_in_loop:
                    delta.data = torch.Tensor(lowpass_firwin(delta.cpu().detach().numpy(), 1024 , fh, fs, window='hamming')).to(device)

                delta = torch.round(torch.clamp(delta, 0, eps), decimals=1)
                batch_delta.grad.data.zero_()
 
    return delta, losses



def UnsyncModelEvaluation(model_name, torch_model, data_loader, delta, gn = False, filter_in_loop= False, fs=100, fh=20):
    classes = (0,1)
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct_pred_model = 0
    total_pred_model   = 0
    pred = []
    target = []
    noise = []
    a_p = []

    for _, (data_test_batched, label_test_batched) in enumerate(data_loader):
        
        if(model_name in ['HTnet', 'SVM']):
            if np.isnan(delta[0].cpu().detach().numpy()):
                delta = torch.zeros_like(delta).to(device)
        else:
            if np.isnan(delta[0].cpu().detach().numpy()):
                delta = torch.zeros_like(delta).to(device)

        
        shift = [randrange(10) for i in range(data_test_batched.shape[0])] 

        if(model_name in ['HTnet', 'SVM']):
            noise = torch.Tensor.repeat(torch.reshape(delta, [1, data_test_batched.shape[1]]), [data_test_batched.shape[0],1]).to(device)
            a_p=torch.stack([torch.concat([noise[i][shift[i]:]
                                   ,noise[i][:shift[i]]]) for i in range(data_test_batched.shape[0])]) 
        else:
            delta_tmp = torch.reshape(delta, [1,2500])
            noise = torch.stack([torch.concat([delta_tmp[i][shift[i]:]
                                   ,delta_tmp[i][:shift[i]]]) for i in range(data_test_batched.shape[0])]) 
            a_p = torch.reshape(noise, [1, 1, 50, 50]).repeat([data_test_batched.shape[0],3,1,1]).to(device)

        trace = torch.stack([data_test_batched[i].to(device) + a_p[i] if torch.argmax(label_test_batched, axis = 1)[i] == 1
              else data_test_batched[i].to(device) for i in range(data_test_batched.shape[0])])

        if gn:
            trace.data = trace + torch.randn(trace.shape).to(device)

        if(model_name in ['HTnet', 'SVM']):  
            if filter_in_loop:
                trace_tmp = [lowpass_firwin(x, 1024 , fh, fs, window='hamming') for x in (trace.cpu().detach().numpy())]
                trace.data = torch.Tensor(trace_tmp).to(device)
        else:
            if filter_in_loop:
                trace_tmp = torch.reshape(trace[0][0], [1, 2500])
                trace_tmp = [lowpass_firwin(x, 1024 , fh, fs, window='hamming') for x in (trace_tmp.cpu().detach().numpy())]
                trace.data= torch.reshape(torch.Tensor(trace_tmp).to(device), [batch_size, 1, 50, 50]).repeat([data_test_batched.shape[0],3,1,1]).to(device)

        if(model_name == 'SVM'):
            pred = torch.argmax(torch_model.predict_proba(trace.type(torch.DoubleTensor).to(device)), axis = 1)
            target  = torch.argmax(label_test_batched, axis = 1) 
        else:
            pred = torch.argmax(torch_model(trace.to(device)), axis = 1)
            target  = torch.argmax(label_test_batched, axis = 1) 

        class_accuracy = []

        for label, prediction in zip(target, pred.cpu().detach().numpy()):
            if label == prediction:
                correct_pred[classes[label]] += 1
                correct_pred_model += 1
            total_pred[classes[label]] += 1
            total_pred_model += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5d} is {accuracy:.1f} %')
        class_accuracy.append(accuracy)

    accuracy_model = 100 * float(correct_pred_model) / total_pred_model
    print(f'Model Accuracy:          {accuracy_model:.1f}%')
    
    return class_accuracy




def SyncPatchPowerBudgetCal(model_name, torch_model , data_loader, nb_epoch=10, resolution=0.1, gn=False, filter_in_loop=False, fs=100, fh=20):
    for eps in np.arange(0, 30, resolution):
        print("eps := ", eps)
        delta , losses = SyncAdversarialPatchGen(model_name, torch_model , data_loader, nb_epoch=nb_epoch, eps=eps, resolution=resolution, gn=gn, filter_in_loop=filter_in_loop, fs=fs, fh=fh)
        class_accuracy = SyncModelEvaluation(model_name, torch_model, data_loader, delta = delta, gn = gn, filter_in_loop= filter_in_loop, fs=fs, fh=fh)
        
        if(class_accuracy[1] == 0):
            break 
            
def UnsyncPatchPowerBudgetCal(model_name, torch_model , data_loader, nb_epoch=10, resolution=0.1, gn=False, filter_in_loop=False, fs=100, fh=20):
    for eps in np.arange(0, 30, resolution):
        print("eps := ", eps)
        delta , losses = UnsyncAdversarialPatchGen(model_name, torch_model , data_loader, nb_epoch=nb_epoch, eps=eps, resolution=resolution, gn=gn, filter_in_loop=filter_in_loop, fs=fs, fh=fh)
        class_accuracy = UnsyncModelEvaluation(model_name, torch_model, data_loader, delta = delta, gn = gn, filter_in_loop= filter_in_loop, fs=fs, fh=fh)
        
        if(class_accuracy[1] == 0):
            break 


def Train_adversarial(model_name, adv_trained_model, x_train_batched, y_train_batched, delta, nb_epoch=1):
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(adv_trained_model.parameters(), lr=1e-11, weight_decay=0.00005)

    if(model_name in ['HTnet', 'SVM']):
        noise = torch.Tensor.repeat(torch.reshape(delta, [1, x_train_batched.shape[1]]), [x_train_batched.shape[0],1]).to(device)

    else:
        noise = torch.reshape(delta, [1,1,delta.shape[0], delta.shape[1]]).repeat([1,3,1,1])#delta#torch.Tensor.repeat(torch.reshape(delta, [1, x_train_batched.shape[1]]), [x_train_batched.shape[0],1]).to(device)

    trace = torch.stack([x_train_batched[i].to(device) + noise[i] if torch.argmax(y_train_batched, axis = 1)[i] == 1
                    else x_train_batched[i].to(device) for i in range(x_train_batched.shape[0])])
    
    xa_val = Variable(trace.to(device))
    ya_val = Variable(y_train_batched.to(device))
    adv_trained_model.train()
    for epoch in range(nb_epoch):  
        if(model_name in ['SVM']):
            outputs_adversarial = adv_trained_model.predict_proba(xa_val.type(torch.DoubleTensor))
        else:
            outputs_adversarial = adv_trained_model(xa_val)
        loss = criterion(outputs_adversarial, ya_val)
        loss.backward()
        optimizer.step()
    
    return adv_trained_model


def AdversarialTraining(model_name, torch_model,  data_loader, batch_size = 20, nb_epoch_noise = 10, eps = 1.2, resolution= 0.1, nb_epochs_at = 2):  

    adv_trained_model = copy.deepcopy(torch_model)

    for epoc in tqdm(range(nb_epochs_at)):
        at_train_deltas = []
        for j, (x_train_batched, y_train_batched) in (enumerate(data_loader)):
            my_dataset_tmp  = TensorDataset(x_train_batched, y_train_batched)
            data_loader_tmp = DataLoader(my_dataset_tmp, batch_size=batch_size)

            at_train_delta, losses = SyncAdversarialPatchGen(model_name, adv_trained_model, data_loader_tmp, nb_epoch_noise, eps, resolution, gn=False, filter_in_loop=False, fs=100, fh=20, at_mode = True)
            adv_trained_model = Train_adversarial(model_name, adv_trained_model, x_train_batched, y_train_batched, at_train_delta, nb_epoch = 1)

    
    return adv_trained_model
    
    


def main():
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-se", "--sync_epsilon", type= float, default= 1.2, help= "Adversarial noise power budget (mW)")
    argParser.add_argument("-ue", "--unsync_epsilon", type= float, default= 12, help= "Adversarial noise power budget (mW)")
    argParser.add_argument("-r", "--resolution", type=float, default= 0.1, help= "Resolution of generated patch")
    argParser.add_argument("-b", "--benchmark", type=str, default='AES-T700', help="Benchmark name")
    argParser.add_argument("-m", "--model_name", type=str, default='HTnet', help="Model name")
    argParser.add_argument("--sync", action='store_true', help="Generate a synchronized patch")   
    argParser.add_argument("--unsync", action='store_true', help="Generate a unsynchronized patch") 
    argParser.add_argument("--at", action='store_true', help="Generate adversarial trained model") 
    argParser.add_argument("-o", "--output_dir", type=str, default='./results/', help="Output directory")
    argParser.add_argument("--gn", action='store_false',  default=False, help="Put a random guasian noise in the power trace") 
    argParser.add_argument("--filter", action='store_false', default=False, help="Activate filter") 
    argParser.add_argument("-fs", "--sample_freq", type= float, default= 100, help= "Filter sample rate")
    argParser.add_argument("-fh", "--high_freq", type= float, default= 20, help= "Filter high frequency")
    argParser.add_argument("-f", "--fff", help="A dummy argument to fool ipython", default="1")



    
    args = argParser.parse_args()
    
    
    benchmark = str(args.benchmark)
    model_name = str(args.model_name) #{'HTnet', 'ResNet-18', 'VGG-11', 'SVM'}
    output_directory = str(args.output_dir)   
    sync_eps = float(args.sync_epsilon)
    unsync_eps = float(args.unsync_epsilon)
    resolution = float(args.resolution)
    sync = bool(args.sync)
    unsync = bool(args.unsync)
    at = bool(args.at)
    
    filter_in_loop = bool(args.filter)
    gn = bool(args.gn)
    fs = float(args.sample_freq)
    fh = float(args.high_freq)
    
    print('Benchmark:=', benchmark)
    print('Model:=', model_name)
    print('Output_directory:=', output_directory)

    if(device.type == 'cuda'):
        print("device := GPU:", torch.cuda.get_device_name(device))
    else:
        print("device := CPU")
        
    if(filter_in_loop):
        print("Sample rate of the filter:= ", fs, "MHz")
        print("High frequency of the filter:= ", fh, "MHz")
    
    nb_epochs = 30
    batch_size = 20
    number_of_samples = 20000
    workers = 16

    if(model_name in ['HTnet', 'SVM']):
        batch_size = 20
    else:
        batch_size = 1
    
    if not os.path.isdir('./dataset/' + benchmark + '_power_Temp25C') or not os.path.isdir('./trained_models/regular_models/pytorch/' + model_name):
        print('**************************************************')
        print('**********Downloading Model and Dataset**********')
    
    Preprocessing(benchmark = benchmark, model_name = model_name)
        
    print('**************************************************')
    print('******************Data is loading******************')
    create_directory(output_directory)
    
    data_train, label_train, data_test, label_test, input_shape, nb_classes = KerasDataPrep(benchmark = 'AES-T700', number_of_samples = number_of_samples, batch_size = batch_size)
    # Training(input_data = data_train, label = label_train,  benchmark = 'AES-T700', output_directory = output_directory, input_shape = input_shape, nb_classes = nb_classes,  batch_size = batch_size, nb_epochs = nb_epochs, workers = workers)

    torch_model = TorchLoadModel(model_name, benchmark)  
    data_loader = TorchDataPrep(model_name, data_test, label_test)
    delta = torch.zeros(data_test.shape[1]).to(device)
    print('******************Data is loaded******************')
    print('**************************************************')

    print('**************************************************')
    print('*****************Model Evaluation*****************')

    ModelEvaluation(model_name, torch_model, data_loader)
    
    if(sync):
        print('**************************************************')
        print('***************Patch is generating****************')

        print('Sync_epsilon:=', sync_eps)
        print('Resolution:=', resolution)

        # SyncPatchPowerBudgetCal(torch_model , data_loader, nb_epoch=10, resolution=resolution, gn=False, filter_in_loop=False, fs=100, fh=20)
        delta , losses = SyncAdversarialPatchGen(model_name, torch_model , data_loader, nb_epoch=nb_epochs, eps=sync_eps, resolution=resolution, gn=gn, filter_in_loop=filter_in_loop, fs=fs, fh=fh)
        SyncModelEvaluation(model_name, torch_model, data_loader, delta = delta, gn = gn, filter_in_loop= filter_in_loop, fs=fs, fh=fh)
        create_directory(output_directory + "patch/" + model_name + '/' + "sync/")
        np.savetxt(output_directory + "patch/" + model_name + '/' + "sync/"  + benchmark + ".txt", delta.cpu().detach().numpy(), fmt='%.2f', delimiter='\0')
        
        print('***************Patch is generated*****************')
        print('**************************************************')

    if(unsync):

        print('**************************************************')
        print('**********Unsynch patch is generating*************')
        
        print('unsync_epsilon:=', unsync_eps)
        print('resolution:=', resolution)

        # UnsyncPatchPowerBudgetCal(torch_model , data_loader, nb_epoch=10, resolution=resolution, gn=False, filter_in_loop=False, fs=100, fh=20)
        delta , losses = UnsyncAdversarialPatchGen(model_name, torch_model , data_loader, nb_epoch=nb_epochs, eps=unsync_eps, resolution=resolution, gn=gn, filter_in_loop=filter_in_loop, fs=fs, fh=fh)
        UnsyncModelEvaluation(model_name, torch_model, data_loader, delta = delta, gn = gn, filter_in_loop= filter_in_loop, fs=fs, fh=fh)

        create_directory(output_directory + "patch/" + model_name + '/' + "unsync/")
        np.savetxt(output_directory + "patch/" + model_name + '/' + "unsync/"  + benchmark + ".txt", delta.cpu().detach().numpy(), fmt='%.2f', delimiter='\0')

        print('**********Unsynch patch is generated*************') 
        print('**************************************************')
        
    if(at):
            print('**************************************************')
            print('********Adversarial training is started***********')

            at_torch_model = AdversarialTraining(model_name, torch_model, data_loader, batch_size = batch_size, nb_epoch_noise = 2, eps = sync_eps, resolution=resolution, nb_epochs_at = 3)    
            at_torch_model.to(device)
            SyncModelEvaluation(model_name, at_torch_model, data_loader, delta = delta, gn = False, filter_in_loop= False, fs=100, fh=20)

            create_directory(output_directory + "models/" + model_name + '/')
            torch.save(at_torch_model, output_directory + "models/" +  model_name + '/' + '/at_' + benchmark + '.pt')

            print('**************Model is generated******************')
            print('**************************************************')

    # SyncPatchPowerBudgetCal(model_name, torch_model, data_loader, nb_epoch=2, resolution=0.1)
if __name__ == '__main__':
    main()
    


# In[ ]:




