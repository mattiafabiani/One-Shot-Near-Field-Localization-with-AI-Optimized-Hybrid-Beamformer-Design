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
        globals()[f'list_r_scat{i}'] = []
        globals()[f'list_theta_scat{i}'] = []
        globals()[f'list_pos_scat{i}'] = []
    temp_r = []
    temp_theta = []
    temp_pos = []
    temp_r_scat = []
    temp_theta_scat = []
    temp_pos_scat = []

    # running_acc = 0.
    running_loss = 0.
    running_size = 0
    with tqdm(iterate_minibatches(X_train, y_train, SNR, batch_size, shuffle=True), unit=' batch', 
              total=int(np.ceil(X_train.shape[0]/batch_size)), file=sys.stdout, leave=True) as tepoch:
        for batch_x, batch_y, batch_SNR in tepoch:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad() # Make the gradients zero
            batch_y_hat = net(batch_x) # Prediction
            
            r, r_pred, r_scat, r_scat_pred, theta, theta_pred, theta_scat, theta_scat_pred, p_true, p_pred, p_scat_true, p_scat_pred = dnn_output_to_near_field_components(batch_y,batch_y_hat,r_lim)
            # print(r.detach().numpy(), r_pred.detach().numpy(), r_scat.detach().numpy(), r_scat_pred.detach().numpy(), theta.detach().numpy(), theta_pred.detach().numpy(), theta_scat.detach().numpy(), theta_scat_pred.detach().numpy(), p_true.detach().numpy(), p_pred.detach().numpy(), p_scat_true.detach().numpy(), p_scat_pred.detach().numpy())
            # exit()
            # calculate results based on SNR
            SNRs = list(set(batch_SNR.cpu().numpy()))
            for idx, snr in enumerate(SNRs):
                mask = (batch_SNR.cpu().numpy() == snr)
                # VERIFY THAT THE SNR IS MASKED CORRECTLY
                # print(((r[mask] - r_pred[mask])**2).cpu().detach().numpy().tolist())
                # print(np.sqrt(np.mean(((r[mask] - r_pred[mask])**2).cpu().detach().numpy().tolist())))
                # exit()
                globals()[f'list_r{idx}'].append(((r[mask] - r_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_theta{idx}'].append(((theta[mask] - theta_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_pos{idx}'].append(torch.sum((p_true[mask] - p_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
                # scatterer's results
                globals()[f'list_r_scat{idx}'].append(((r_scat[mask] - r_scat_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_theta_scat{idx}'].append(((theta_scat[mask] - theta_scat_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_pos_scat{idx}'].append(torch.sum((p_scat_true[mask] - p_scat_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
            
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
        temp_r.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_r{i}'])))))
        temp_theta.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_theta{i}'])))))
        temp_pos.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos{i}'])))))
        # scatterer's results
        temp_r_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_r_scat{i}'])))))
        temp_theta_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_theta_scat{i}'])))))
        temp_pos_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos_scat{i}'])))))

    RMSE = {'r': temp_r, 'theta': temp_theta, 'pos': temp_pos,'r_scat': temp_r_scat, 'theta_scat': temp_theta_scat, 'pos_scat': temp_pos_scat}
    return curr_loss, RMSE
    
def eval_loop(X_val, y_val, SNR, net, criterion, device,N,N_RF,r_lim, type, batch_size=64):
    net.eval()
    for i in SNR.unique().tolist():
        globals()[f'list_r{i}'] = []
        globals()[f'list_theta{i}'] = []
        globals()[f'list_pos{i}'] = []
        globals()[f'list_r_scat{i}'] = []
        globals()[f'list_theta_scat{i}'] = []
        globals()[f'list_pos_scat{i}'] = []
    temp_r = []
    temp_theta = []
    temp_pos = []
    temp_r_scat = []
    temp_theta_scat = []
    temp_pos_scat = []
    
    # running_acc = 0.0
    running_loss = 0.0
    running_size = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_SNR in iterate_minibatches(X_val, y_val, SNR, batch_size, shuffle=False):
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y_hat = net(batch_x) # Prediction

            
            r, r_pred, r_scat, r_scat_pred, theta, theta_pred, theta_scat, theta_scat_pred, p_true, p_pred, p_scat_true, p_scat_pred = dnn_output_to_near_field_components(batch_y,batch_y_hat,r_lim)

            # calculate results based on SNR
            SNRs = list(set(batch_SNR.cpu().numpy()))
            for idx, snr in enumerate(SNRs):
                mask = (batch_SNR == snr)
                globals()[f'list_r{idx}'].append(((r[mask] - r_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_theta{idx}'].append(((theta[mask] - theta_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_pos{idx}'].append(torch.sum((p_true[mask] - p_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
                # scatterer's results
                globals()[f'list_r_scat{idx}'].append(((r_scat[mask] - r_scat_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_theta_scat{idx}'].append(((theta_scat[mask] - theta_scat_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_pos_scat{idx}'].append(torch.sum((p_scat_true[mask] - p_scat_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
            
            loss = criterion(batch_y_hat, batch_y) # Loss computation
            running_loss += loss.item() * batch_x.shape[0]
            running_size += batch_x.shape[0]
            curr_loss = running_loss/running_size
    # print(SNR.unique().tolist())
    # exit()
    for i in SNR.unique().tolist():
        temp_r.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_r{i}'])))))
        temp_theta.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_theta{i}'])))))
        temp_pos.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos{i}'])))))
        # scatterer's results
        temp_r_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_r_scat{i}'])))))
        temp_theta_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_theta_scat{i}'])))))
        temp_pos_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos_scat{i}'])))))

    RMSE = {'r': temp_r, 'theta': temp_theta, 'pos': temp_pos,'r_scat': temp_r_scat, 'theta_scat': temp_theta_scat, 'pos_scat': temp_pos_scat}
    return curr_loss, RMSE


def test_loop(X_test, y_test, SNR, net, criterion, device,N,N_RF,r_lim, type, batch_size=64):
    net.eval()
    for i in SNR.unique().tolist():
        globals()[f'list_r{i}'] = []
        globals()[f'list_theta{i}'] = []
        globals()[f'list_pos{i}'] = []
        globals()[f'list_r_scat{i}'] = []
        globals()[f'list_theta_scat{i}'] = []
        globals()[f'list_pos_scat{i}'] = []
    temp_r = []
    temp_theta = []
    temp_pos = []
    temp_r_scat = []
    temp_theta_scat = []
    temp_pos_scat = []

    snr_list = []
    r_pred_list = []
    r_true_list = []
    theta_pred_list = []
    theta_true_list = []
    range_error_list = []
    theta_error_list = []
    pos_error_list = []
    range_scat_error_list = []
    theta_scat_error_list = []
    pos_scat_error_list = []
    
    # running_acc = 0.0
    running_loss = 0.0
    running_size = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_SNR in iterate_minibatches(X_test, y_test, SNR, batch_size, shuffle=False):
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y_hat = net(batch_x) # Prediction

            
            r, r_pred, r_scat, r_scat_pred, theta, theta_pred, theta_scat, theta_scat_pred, p_true, p_pred, p_scat_true, p_scat_pred = dnn_output_to_near_field_components(batch_y,batch_y_hat,r_lim)

            # calculate results based on SNR
            SNRs = list(set(batch_SNR.cpu().numpy()))
            for idx, snr in enumerate(SNRs):
                mask = (batch_SNR == snr)
                globals()[f'list_r{idx}'].append(((r[mask] - r_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_theta{idx}'].append(((theta[mask] - theta_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_pos{idx}'].append(torch.sum((p_true[mask] - p_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
                # scatterer's results
                globals()[f'list_r_scat{idx}'].append(((r_scat[mask] - r_scat_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_theta_scat{idx}'].append(((theta_scat[mask] - theta_scat_pred[mask])**2).cpu().detach().numpy().tolist())
                globals()[f'list_pos_scat{idx}'].append(torch.sum((p_scat_true[mask] - p_scat_pred[mask])**2, dim=1).cpu().detach().numpy().tolist())
            
            
            for i in range(len(batch_SNR)):
                snr_list.append(batch_SNR[i].cpu().item()*5)
                r_pred_list.append(r_pred[i].cpu().item())
                r_true_list.append(r[i].cpu().item())
                theta_pred_list.append(theta_pred[i].cpu().item())
                theta_true_list.append(theta[i].cpu().item())
                range_error_list.append(((r[i] - r_pred[i])**2).cpu().item())
                theta_error_list.append(((theta[i] - theta_pred[i])**2).cpu().item())
                pos_error_list.append(torch.sum((p_true[i] - p_pred[i])**2).cpu().item())
                range_scat_error_list.append(((r_scat[i] - r_scat_pred[i])**2).cpu().item())
                theta_scat_error_list.append(((theta_scat[i] - theta_scat_pred[i])**2).cpu().item())
                pos_scat_error_list.append(torch.sum((p_scat_true[i] - p_scat_pred[i])**2).cpu().item())
                # if batch_SNR[i].cpu().item()*5 == 10:
                # if i == 9:
                #     print(f'r_pred: {r_pred[i]:.1f}, r_true: {r[i]:.1f} (RMSE {(r[i] - r_pred[i])**2:.1f}), theta_pred: {theta_pred[i]:.1f}, theta_true: {theta[i]:.1f} (RMSE {(theta[i] - theta_pred[i])**2:.1f}),')
                #     print(f'r_scat_pred: {r_scat_pred[i]:.1f}, r_scat_true: {r_scat[i]:.1f} (RMSE {(r_scat[i] - r_scat_pred[i])**2:.1f}), theta_scat_pred: {theta_scat_pred[i]:.1f}, theta_scat_true: {theta_scat[i]:.1f} (RMSE {(theta_scat[i] - theta_scat_pred[i])**2:.1f}), ')
                #     print(f'pos_pred: {p_pred[i]}, pos_true: {p_true[i]}, pos_scat_pred: {p_scat_pred[i]}, pos_scat_true: {p_scat_true[i]}, ')
                #     print(f'rmse pos: {torch.sum((p_true[i] - p_pred[i])**2).cpu().item()}, rmse pos_scat: {torch.sum((p_scat_true[i] - p_scat_pred[i])**2).cpu().item()}')
                #     import matplotlib.pyplot as plt
                #     plt.plot(p_pred[i][0],p_pred[i][1],'or',label='p_pred')
                #     plt.plot(p_true[i][0],p_true[i][1],'ok',label='p_true')
                #     plt.plot(p_scat_pred[i][0],p_scat_pred[i][1],'xr',label='p_scat_pred')
                #     plt.plot(p_scat_true[i][0],p_scat_true[i][1],'xk',label='p_scat_true')
                #     plt.ylim([-10, 10])
                #     plt.xlim([0, 20])
                #     # plt.axis('equal')
                #     plt.legend()
                #     plt.show()
                #     exit()
            
            loss = criterion(batch_y_hat, batch_y) # Loss computation
            running_loss += loss.item() * batch_x.shape[0]
            running_size += batch_x.shape[0]
            curr_loss = running_loss/running_size
    # print(SNR.unique().tolist())
    # exit()
    for i in SNR.unique().tolist():
        temp_r.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_r{i}'])))))
        temp_theta.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_theta{i}'])))))
        temp_pos.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos{i}'])))))
        # scatterer's results
        temp_r_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_r_scat{i}'])))))
        temp_theta_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_theta_scat{i}'])))))
        temp_pos_scat.append(np.sqrt(np.mean(list(itertools.chain.from_iterable(globals()[f'list_pos_scat{i}'])))))

    RMSE = {'r': temp_r, 'theta': temp_theta, 'pos': temp_pos,'r_scat': temp_r_scat, 'theta_scat': temp_theta_scat, 'pos_scat': temp_pos_scat}
    data = {
            'SNR': snr_list,
            'r_pred': r_pred_list,
            'r_true': r_true_list,
            'theta_pred': theta_pred_list,
            'theta_true': theta_true_list,
            'Test (r)': range_error_list,
            'Test (theta)': theta_error_list,
            'Test (pos)': pos_error_list,
            'Test (r_scat)': range_scat_error_list,
            'Test (theta_scat)': theta_scat_error_list,
            'Test (pos_scat)': pos_scat_error_list,
        }
    return curr_loss, RMSE, data

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

def dnn_output_to_near_field_components(batch_y,batch_y_hat,r_lim):
    # convert DNN output to range and angle
    [r_pred, _, theta_pred] = range_angle_from_net_output(batch_y_hat[:,[0,2]],r_lim)
    [r, _, theta] = range_angle_from_net_output(batch_y[:,[0,2]],r_lim)
    [r_scat_pred, _, theta_scat_pred] = range_angle_from_net_output(batch_y_hat[:,[1,3]],r_lim) # multipath component
    [r_scat, _, theta_scat] = range_angle_from_net_output(batch_y[:,[1,3]],r_lim)

    # from polar coordinate to cartesian coordinates (near-field source + near-field scatterer)
    sin_theta = batch_y[:,0]
    sin_theta_scat = batch_y[:,1]
    sin_theta_pred = batch_y_hat[:,0]
    sin_theta_scat_pred = batch_y_hat[:,1]
    p_true = torch.stack((r*torch.sqrt(1-sin_theta**2),r*sin_theta),axis=-1)
    p_scat_true = torch.stack((r_scat*torch.sqrt(1-sin_theta_scat**2),r_scat*sin_theta_scat),axis=-1) # x = r*cos(theta), y = r*sin(theta)
    p_pred = torch.stack((r*torch.sqrt(1-sin_theta_pred**2),r_pred*sin_theta_pred),axis=-1)
    p_scat_pred = torch.stack((r_scat_pred*torch.sqrt(1-sin_theta_scat_pred**2),r_scat_pred*sin_theta_scat_pred),axis=-1)
    return r, r_pred, r_scat, r_scat_pred, theta, theta_pred, theta_scat, theta_scat_pred, p_true, p_pred, p_scat_true, p_scat_pred