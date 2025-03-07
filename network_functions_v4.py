import torch
from tqdm import tqdm
import numpy as np
import sys
import os
import pandas as pd
import itertools
from dnn_model import weightConstraint

def train_loop(X_train, y_train, SNR, net, optimizer, criterion, device,N,N_RF,x_lim,y_lim, type, batch_size=64):
    net.train()

    for i in SNR.unique().tolist():
        globals()[f'list_pos{i}'] = []
    temp_pos = []

    running_loss = 0.
    running_size = 0
    with tqdm(iterate_minibatches(X_train, y_train, SNR, batch_size, shuffle=True), unit=' batch', 
              total=int(np.ceil(X_train.shape[0]/batch_size)), file=sys.stdout, leave=True) as tepoch:
        for batch_x, batch_y, batch_SNR in tepoch:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad() # Make the gradients zero

            # print(batch_x.size())
            batch_y_hat = net(batch_x) # Prediction

            x_pred = x_lim[0] + batch_y_hat[:,0]*(x_lim[1] - x_lim[0]) # de-scaling
            y_pred = y_lim[0] + batch_y_hat[:,1]*(y_lim[1] - y_lim[0])
            p_pred = torch.stack((x_pred,y_pred),axis=-1)
            
            x = x_lim[0] + batch_y[:,0]*(x_lim[1] - x_lim[0]) # de-scaling
            y = y_lim[0] + batch_y[:,1]*(y_lim[1] - y_lim[0])
            p_true = torch.stack((x,y),axis=-1)

            # calculate results based on SNR
            SNRs = list(set(batch_SNR.cpu().numpy()))
            for idx, snr in enumerate(SNRs):
                mask = (batch_SNR.cpu().numpy() == snr)
                globals()[f'list_pos{idx}'].append(torch.sum((p_true[mask] - p_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
                
            loss = criterion(batch_y_hat, batch_y) # Loss computation
            loss.backward() # Backward step
            optimizer.step() # Update coefficients
            
            constraints = weightConstraint(N,N_RF,type) # per-antenna power constraint (eq 5d)
            net._modules['fc1'].apply(constraints)
            
            
            running_loss += loss.item() * batch_x.shape[0]
            running_size += batch_x.shape[0]
            
            curr_loss = running_loss/running_size
            tepoch.set_postfix(loss=curr_loss)

        curr_loss = running_loss/len(X_train)

    for i in SNR.unique().tolist():
        temp_pos.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos{i}'])))))

    RMSE = {'pos': temp_pos}
    return curr_loss, RMSE
    
def eval_loop(X_val, y_val, SNR, net, criterion, device,N,N_RF,x_lim,y_lim, type, batch_size=64):
    net.eval()
    for i in SNR.unique().tolist():
        globals()[f'list_pos{i}'] = []
    temp_r = []
    temp_theta = []
    temp_pos = []
    
    # running_acc = 0.0
    running_loss = 0.0
    running_size = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_SNR in iterate_minibatches(X_val, y_val, SNR, batch_size, shuffle=False):
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_y_hat = net(batch_x) # Prediction

            x_pred = x_lim[0] + batch_y_hat[:,0]*(x_lim[1] - x_lim[0]) # de-scaling
            y_pred = y_lim[0] + batch_y_hat[:,1]*(y_lim[1] - y_lim[0])
            p_pred = torch.stack((x_pred,y_pred),axis=-1)
            
            x = x_lim[0] + batch_y[:,0]*(x_lim[1] - x_lim[0]) # de-scaling
            y = y_lim[0] + batch_y[:,1]*(y_lim[1] - y_lim[0])
            p_true = torch.stack((x,y),axis=-1)

            # calculate results based on SNR
            SNRs = list(set(batch_SNR.cpu().numpy()))
            for idx, snr in enumerate(SNRs):
                mask = (batch_SNR.cpu().numpy() == snr)
                globals()[f'list_pos{idx}'].append(torch.sum((p_true[mask] - p_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())

            # print(f'\npred: {batch_y_hat}, true: {batch_y}')
            # print(f'\np_pred: {p_pred}, p_true: {p_true}')
            # exit()
            loss = criterion(batch_y_hat, batch_y) # Loss computation
            running_loss += loss.item() * batch_x.shape[0]
            running_size += batch_x.shape[0]
            curr_loss = running_loss/running_size

    for i in SNR.unique().tolist():
        temp_pos.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos{i}'])))))

    RMSE = {'pos': temp_pos}
    return curr_loss, RMSE
    
def test_loop(X_test, y_test, net, device,N,N_RF,x_lim,y_lim, model_path=None):
    # Network Setup
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    net.eval()
    
    # Timers
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_time = 0.
    running_rmse_r = 0.
    running_rmse_theta = 0.
    running_rmse_pos = 0.
    running_size = 0.
    
    #GPU-WARM-UP
    data_shape_single = list(X_test.shape)
    data_shape_single[0] = 1
    data_shape_single = tuple(data_shape_single)
    with torch.no_grad():
        for _ in range(10):
            dummy_input = torch.randn(data_shape_single, dtype=torch.float, device=device)
            out = net(dummy_input)
            
    y = -1*torch.ones(len(X_test))
    y_hat = -1*torch.ones((len(X_test), out.shape[1])) #Top 5
    
    cur_ind = 0
    
    # Test
    with torch.no_grad():
        for batch_x, batch_y in iterate_minibatches(X_test, y_test, batchsize=1, shuffle=False):
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            #### Measure Inference Duration ####
            # starter.record()
            
            batch_y_hat = net(batch_x) # Prediction

            x_pred = x_lim[0] + batch_y_hat[:,0]*(x_lim[1] - x_lim[0]) # de-scaling
            y_pred = y_lim[0] + batch_y_hat[:,0]*(y_lim[1] - y_lim[0])
            p_pred = torch.stack((x_pred,y_pred),axis=-1)
            
            x = x_lim[0] + batch_y[:,0]*(x_lim[1] - x_lim[0]) # de-scaling
            y = y_lim[0] + batch_y[:,0]*(y_lim[1] - y_lim[0])
            p_true = torch.stack((x,y),axis=-1)
            curr_rmse_pos = torch.sqrt(torch.mean(torch.sum((p_true - p_pred)**2, dim=1)))
    
            # predictions = batch_y_hat.argmax(dim=1, keepdim=True).squeeze()
            # running_acc += (predictions == batch_y).sum().item()
            # running_loss += criterion(batch_y_hat, batch_y).item() * batch_x.shape[0]
            running_rmse_pos += curr_rmse_pos * batch_x.shape[0]
            running_size += batch_x.shape[0]
            
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)/1000
            # total_time += curr_time
            #####################################
            
            ###### Save Top-k Predictions #######
            # next_ind = cur_ind + batch_x.shape[0]
            # y[cur_ind:next_ind] = batch_y
            # y_hat[cur_ind:next_ind, :] = batch_y_hat
            # cur_ind = next_ind
            #####################################

    # network_time_per_sample = total_time / len(X_test)
    
    return curr_rmse_pos
    

def evaluate_predictions(y, y_hat, k):
    topk_pred = torch.topk(y_hat, k=k).indices
    topk_acc = np.zeros(k)
    for i in range(k):
        topk_acc[i] = torch.mean((y == topk_pred[:, i])*1.0)
    topk_acc = np.cumsum(topk_acc)
    
    beam_dist = torch.mean(torch.abs(y - topk_pred[:, 0]))
    
    return topk_acc, beam_dist

def iterate_minibatches(X, y, SNR, batchsize, shuffle=False):
    
    data_len = X.shape[0]
    indices = np.arange(data_len)
    if shuffle:
        np.random.shuffle(indices)
        
    for start_idx in range(0, data_len, batchsize):
        end_idx = min(start_idx + batchsize, data_len)
        excerpt = indices[start_idx:end_idx]
        yield X[excerpt], y[excerpt], SNR[excerpt]
        
def range_angle_from_net_output(y_hat,r_lim):
    r = r_lim[0] + (y_hat[:,1] + 1)/2 * (r_lim[1] - r_lim[0])
    angle_rad = torch.asin(y_hat[:,0])
    angle_deg = angle_rad * 180/torch.math.pi
    return [r, angle_rad, angle_deg]

def pol2cart(r,theta):
    '''
    input: array of
        - r: range
        - theta: angle in degrees
    output: array of
        - pos: cartesian position
    '''
    x = r * torch.cos(torch.deg2rad(theta))
    y = r * torch.sin(torch.deg2rad(theta))
    pos = torch.stack((x, y), dim=1) # shape (batch,2)
    return pos