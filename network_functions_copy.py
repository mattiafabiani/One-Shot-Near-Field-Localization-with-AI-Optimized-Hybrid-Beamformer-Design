import torch
from tqdm import tqdm
import numpy as np
import sys
import os
import pandas as pd
import itertools
from dnn_model import weightConstraint

def train_loop(X_train, y_train, SNR, net, optimizer, criterion, device,N,N_RF,r_lim, type, batch_size=64):
    net.train()

    for i in SNR.unique().tolist():
        globals()[f'list_r{i}'] = []
        globals()[f'list_theta{i}'] = []
        globals()[f'list_pos{i}'] = []
    temp_r = []
    temp_theta = []
    temp_pos = []

    # running_acc = 0.
    running_loss = 0.
    running_size = 0
    with tqdm(iterate_minibatches(X_train, y_train, SNR, batch_size, shuffle=True), unit=' batch', 
              total=int(np.ceil(X_train.shape[0]/batch_size)), file=sys.stdout, leave=True) as tepoch:
        for batch_x, batch_y, batch_SNR in tepoch:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad() # Make the gradients zero

            # print(batch_x.size())
            batch_y_hat = net(batch_x) # Prediction
            # print(batch_y_hat.size())
            # exit()
            
            [r_pred, _, theta_pred] = range_angle_from_net_output(batch_y_hat,r_lim)
            [r, _, theta] = range_angle_from_net_output(batch_y,r_lim)
            sin_theta = batch_y[:,0]
            x, y = r*sin_theta, r*np.sqrt(1-sin_theta**2)
            p_true = torch.stack((x,y),axis=-1)
            sin_theta = batch_y_hat[:,0]
            x, y = r_pred.detach()*sin_theta.detach(), r_pred.detach()*np.sqrt(1-sin_theta.detach()**2)
            p_pred = torch.stack((x,y),axis=-1)

            # calculate results based on SNR
            SNRs = list(set(batch_SNR.cpu().numpy()))
            for idx, snr in enumerate(SNRs):
                mask = (batch_SNR.cpu().numpy() == snr)
                globals()[f'list_r{idx}'].append(((r[mask] - r_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_theta{idx}'].append(((theta[mask] - theta_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_pos{idx}'].append(torch.sum((p_true[mask] - p_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
            
            loss = criterion(batch_y_hat, batch_y) # Loss computation
            # for name, params in net.named_parameters():
            #     if 'weight' in name and 'fc1' in name:
            #         print('Weights fc1 BEFORE weights update')
            #         w = params[:N_RF/2,:N] + 1j*params[N_RF/2:N_RF,:N]
            #         sum_weights = np.sum(np.abs(w.detach().numpy())**2)
            #         print(sum_weights)
            loss.backward() # Backward step
            optimizer.step() # Update coefficients
            
            constraints = weightConstraint(N,N_RF,type) # per-antenna power constraint (eq 5d)
            net._modules['fc1'].apply(constraints)
            
            # for name, params in net.named_parameters():
            #     if 'weight' in name and 'fc1' in name:
            #         print('Weights fc1 AFTER weights update')
            #         print(params.shape)
            #         w = params[:N_RF/2,:N] + 1j*params[N_RF/2:N_RF,:N]
            #         sum_weights = np.sum(np.abs(w.detach().numpy())**2)
            #         print(sum_weights)
            # exit()
            
            # running_acc += (predictions == batch_y).sum().item()
            running_loss += loss.item() * batch_x.shape[0]
            running_size += batch_x.shape[0]
            # running_rmse_r += curr_rmse_r * batch_x.shape[0]
            # running_rmse_theta += curr_rmse_theta * batch_x.shape[0]
            # running_rmse_pos += curr_rmse_pos * batch_x.shape[0]
            # curr_acc = 100. * running_acc/running_size
            curr_loss = running_loss/running_size
            # curr_rmse_pos = running_rmse_pos/running_size
            
            # tepoch.set_postfix(loss=curr_loss, accuracy=curr_acc)
            tepoch.set_postfix(loss=curr_loss)
            
        # curr_acc = 100. * running_acc/len(X_train)
        # curr_loss = running_loss/np.ceil(X_train.shape[0]/batch_size)
        curr_loss = running_loss/len(X_train)

    for i in SNR.unique().tolist():
        temp_r.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_r{i}'])))))
        temp_theta.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_theta{i}'])))))
        temp_pos.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos{i}'])))))

    RMSE = {'r': temp_r, 'theta': temp_theta, 'pos': temp_pos}
    return curr_loss, RMSE
    
def eval_loop(X_val, y_val, SNR, net, criterion, device,N,N_RF,r_lim, type, batch_size=64):
    net.eval()
    for i in SNR.unique().tolist():
        globals()[f'list_r{i}'] = []
        globals()[f'list_theta{i}'] = []
        globals()[f'list_pos{i}'] = []
    temp_r = []
    temp_theta = []
    temp_pos = []
    
    # running_acc = 0.0
    running_loss = 0.0
    running_size = 0
    # time_inference = []
    # kkk = 0
    # import time
    with torch.no_grad():
        for batch_x, batch_y, batch_SNR in iterate_minibatches(X_val, y_val, SNR, batch_size, shuffle=False):
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y_hat = net(batch_x) # Prediction

            [r_pred, _, theta_pred] = range_angle_from_net_output(batch_y_hat,r_lim)
            [r, _, theta] = range_angle_from_net_output(batch_y,r_lim)
            sin_theta = batch_y[:,0]
            x, y = r*sin_theta, r*np.sqrt(1-sin_theta**2)
            p_true = torch.stack((x,y),axis=-1)
            sin_theta = batch_y_hat[:,0]
            x, y = r_pred*sin_theta, r_pred*np.sqrt(1-sin_theta**2)
            p_pred = torch.stack((x,y),axis=-1)
            

            # import matplotlib.pyplot as plt
            # plt.scatter(p_pred.cpu().numpy()[:, 0], p_pred.cpu().numpy()[:, 1], c='k', label='Predicted')  # p_pred scatter (black points)
            # plt.scatter(p_true.cpu().numpy()[:, 0], p_true.cpu().numpy()[:, 1], c='r', label='True')  # p_true scatter (red points)
            # for i in range(len(p_pred.cpu().numpy())):
            #     plt.text(p_pred.cpu().numpy()[i, 0], p_pred.cpu().numpy()[i, 1], str(i+1), fontsize=9, color='k') # add number for each point
            #     plt.text(p_true.cpu().numpy()[i, 0], p_true.cpu().numpy()[i, 1], str(i+1), fontsize=9, color='r')
            #     plt.plot([p_pred.cpu().numpy()[i, 0], p_true.cpu().numpy()[i, 0]], [p_pred.cpu().numpy()[i, 1], p_true.cpu().numpy()[i, 1]], 'k--',linewidth=.5) # connect p_pred and p_true with a dashed line
            # plt.xlim([-10,10])
            # plt.ylim([0,10])
            # plt.grid()
            # plt.axis('equal')
            # plt.legend()
            # plt.show()
            # exit()

            # calculate results based on SNR
            SNRs = list(set(batch_SNR.cpu().numpy()))
            for idx, snr in enumerate(SNRs):
                mask = (batch_SNR == snr)
                globals()[f'list_r{idx}'].append(((r[mask] - r_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_theta{idx}'].append(((theta[mask] - theta_pred[mask])**2).cpu().detach().numpy().tolist())
                # globals()[f'list_pos{idx}'].append(torch.norm(p_true[mask] - p_pred[mask],dim=1).cpu().detach().numpy().tolist())
                globals()[f'list_pos{idx}'].append(torch.sum((p_true[mask] - p_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
            
            # curr_rmse_r = torch.sqrt(torch.mean((r - r_pred)**2))
            # curr_rmse_theta = torch.sqrt(torch.mean((batch_y[:,0] - batch_y_hat[:,1])**2))
            # curr_rmse_pos = torch.sqrt(torch.mean(torch.sum((p_true - p_pred)**2, dim=1)))
            
            loss = criterion(batch_y_hat, batch_y) # Loss computation
    
            # constraints = weightConstraint(N,N_RF,type) # per-antenna power constraint (eq 5d)
            # net._modules['fc1'].apply(constraints)

            # predictions = batch_y_hat.argmax(dim=1, keepdim=True).squeeze()
            # running_acc += (predictions == batch_y).sum().item()
            # running_loss += criterion(batch_y_hat, batch_y).item() * batch_x.shape[0]
            running_loss += loss.item() * batch_x.shape[0]
            # running_rmse_r += curr_rmse_r * batch_x.shape[0]
            # running_rmse_theta += curr_rmse_theta * batch_x.shape[0]
            # running_rmse_pos += curr_rmse_pos * batch_x.shape[0]
            running_size += batch_x.shape[0]
            curr_loss = running_loss/running_size

    for i in SNR.unique().tolist():
        temp_r.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_r{i}'])))))
        temp_theta.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_theta{i}'])))))
        temp_pos.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos{i}'])))))

    RMSE = {'r': temp_r, 'theta': temp_theta, 'pos': temp_pos}
    return curr_loss, RMSE
    
def test_loop(X_test, y_test, SNR, net, criterion, device,N,N_RF,r_lim, type, batch_size=64):
    net.eval()
    for i in SNR.unique().tolist():
        globals()[f'list_r{i}'] = []
        globals()[f'list_theta{i}'] = []
        globals()[f'list_pos{i}'] = []

    snr_list = []
    r_pred_list = []
    r_true_list = []
    theta_pred_list = []
    theta_true_list = []
    range_error_list = []
    theta_error_list = []
    pos_error_list = []
    
    # running_acc = 0.0
    running_loss = 0.0
    running_size = 0
    # time_inference = []
    # kkk = 0
    # import time
    with torch.no_grad():
        for batch_x, batch_y, batch_SNR in iterate_minibatches(X_test, y_test, SNR, batch_size, shuffle=False):
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y_hat = net(batch_x) # Prediction

            ## Code to measure inference time
            # start = time.perf_counter()
            # batch_y_hat = net(batch_x) # Prediction
            # if kkk > 200:
            #     time_inference.append((time.perf_counter() - start)*1000)
            # # print((time.time() - start)*1000)
            # kkk = kkk + 1
            # if kkk == 1200:
            #     print('average_time: ')
            #     print(np.mean(time_inference))
            #     import matplotlib.pyplot as plt
            #     plt.stem(time_inference)
            #     plt.show()
            #     exit()
            # continue

            [r_pred, _, theta_pred] = range_angle_from_net_output(batch_y_hat,r_lim)
            [r, _, theta] = range_angle_from_net_output(batch_y,r_lim)
            sin_theta = batch_y[:,0]
            x, y = r*sin_theta, r*np.sqrt(1-sin_theta**2)
            p_true = torch.stack((x,y),axis=-1)
            sin_theta = batch_y_hat[:,0]
            x, y = r_pred*sin_theta, r_pred*np.sqrt(1-sin_theta**2)
            p_pred = torch.stack((x,y),axis=-1)

            # calculate results based on SNR
            for i in range(len(batch_SNR)):
                snr_list.append(batch_SNR[i].cpu().item()*5)
                r_pred_list.append(r_pred[i].cpu().item())
                r_true_list.append(r[i].cpu().item())
                theta_pred_list.append(theta_pred[i].cpu().item())
                theta_true_list.append(theta[i].cpu().item())
                range_error_list.append(((r[i] - r_pred[i])**2).cpu().item())
                theta_error_list.append(((theta[i] - theta_pred[i])**2).cpu().item())
                pos_error_list.append(torch.sum((p_true[i] - p_pred[i])**2, dim=1).cpu().item())

                # if theta[i] > 0:
                #     print(theta[i])
                #     print(p_pred[i],p_true[i])
                #     print(torch.norm(p_pred[i] - p_true[i]))
                #     exit()
            
            loss = criterion(batch_y_hat, batch_y) # Loss computation
            running_loss += loss.item() * batch_x.shape[0]
            running_size += batch_x.shape[0]
            curr_loss = running_loss/running_size
        data = {
            'SNR': snr_list,
            'r_pred': r_pred_list,
            'r_true': r_true_list,
            'theta_pred': theta_pred_list,
            'theta_true': theta_true_list,
            'Test (r)': range_error_list,
            'Test (theta)': theta_error_list,
            'Test (pos)': pos_error_list
        }
        return curr_loss, data
    

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