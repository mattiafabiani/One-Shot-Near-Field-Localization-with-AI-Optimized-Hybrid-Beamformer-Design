'''
Author: Mattia Fabiani
Update: This version of the main allows the grouping of the results by SNR values.

To generate the dataset run this in the command line: python main_v2_copy.py --generate_dataset 1 --dataset_size 20000 --dataset_name my_dataset
To train the model type these:                        python main_v2_copy.py --train 1 --dataset_name TIMES --dataset_size 20000 --epochs 50 --type sub-connected --logdir saved_models/single_user_TIMES/N_RF  --batch_size 250 --lr 0.003 --N 128 --N_RF 64 --model 1
                                                      '''

import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import argparse
import torch.optim as optim
import os
from dnn_model import CNN_model, DNN_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from network_functions import train_loop, eval_loop, test_loop

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,                   default='', help='Unique experiment identifier')
parser.add_argument('--epochs', type=int,               default=50, help='Number of train epochs.')
parser.add_argument('--N', type=int,                    default=128, help='Number of antennas.')
parser.add_argument('--N_RF', type=int,                 default=16, help='Number of RF chains.')
parser.add_argument('--type', type=str,                 default='sub-connected', help='RF chains to antenna connection')
parser.add_argument('--model', type=int,                default=1, help='Specify model') # 0: DNN, 1: CNN
parser.add_argument('--generate_dataset', type=int,     default=0, help='Generate Dataset.')
parser.add_argument('--dataset_size', type=int,         default=20000, help='# of samples for each SNR value.')
parser.add_argument('--dataset_name', type=str,         default='polar_uniform', help='Default dataset name')
parser.add_argument('--train', type=int,                default=1, help='Train the DNN model.')
parser.add_argument('--lr', type=float,                 default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int,           default=256, help='Batch size')
parser.add_argument('--train_split', type=float,        default=0.8, help='Train split')
parser.add_argument('--logdir', type=str,               default='saved_models', help='Directory to log data to')
parser.add_argument('--scheduler', type=int,            default=0, help='use scheduler to control the learning rate')
args = parser.parse_args()
foldername = args.type + '_' +'epochs'+str(args.epochs)+'_batch'+str(args.batch_size)+'_lr'+str(args.lr)+'_'+str(args.N_RF)+'RF_'+str(args.N)+'N'
args.logdir = os.path.join(args.logdir, foldername + '_' + args.id)

#---------- SIMULATION PARAMETERS -----------
f0 = 25e9                   # carrier frequency
k = 2*np.pi / (3e8 / f0)    # wave number
wavelength = 3e8/f0
d = wavelength / 2              # antenna spacing
N = args.N                     # antennas
N_RF = args.N_RF                   # RF chains
SNR_dB = list(range(0,25,5))
SNR = [10 ** (SNR / 10) for SNR in SNR_dB]

range_limits = [1, 10]      # near-field range limits [m]


array_length = d * (N - 1)
near_field_boundary = 2 * array_length ** 2 / wavelength
print(f'Near-field up to {near_field_boundary:.1f} m'
      f'\nArray length: {array_length*100} cm,\nwavelength = {wavelength*1000} mm')

dataset_size = args.dataset_size        # number of signals per SNR (10k * 5 = 50k samples in ottal)
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
train_val_split = args.train_split
val_split = 0.1
test_split = 0.1
dataset_root = 'dataset/'
dataset_name = args.dataset_name+'_'+str(int(dataset_size*len(SNR)/1e3)) + 'k.npy'
dataset_path = os.path.join(dataset_root,dataset_name)
models = [DNN_model, CNN_model]
rng_seed = 42
#------------------------------------------

#%% Data Prep

if args.generate_dataset:
    def CN_realization(mean, std_dev, size=1):
        return np.random.normal(mean, std_dev, size) + 1j * np.random.normal(mean, std_dev, size)

    delta = lambda n: (2*n - N + 1)/2

    # near-field array response vector (parabolic wavefront approximation)
    a = lambda theta, r: np.array([np.exp(-1j*k*(np.sqrt(r**2 + delta(n)**2*d**2 - 2*r*theta*delta(n)*d) - r)) for n in range(N)]).T

    dataset = dict()


    np.random.seed(rng_seed)
    ii = 0
    s = 1
    for i in tqdm(range(dataset_size)):
        for snr in SNR:
            sigma_n = 1 / np.sqrt(snr)

            p = np.random.uniform(low=0,high=2*range_limits[1],size=(2,)) - [range_limits[1],0]
            r =  np.linalg.norm(p)
            while r < range_limits[0] or r > range_limits[1]:
                p = np.random.uniform(low=0,high=2*range_limits[1],size=(2,)) - [range_limits[1],0]
                r =  np.linalg.norm(p)
            theta = np.sin(np.pi/2 - np.arctan2(p[1],p[0]))
            r_norm = 2 * (r - range_limits[0]) / (range_limits[1] - range_limits[0]) - 1

            n = CN_realization(mean=0, std_dev=sigma_n, size=N)
            # uplink received signal
            y_ = a(theta,r) * s + n
            y = np.concatenate((y_.real, y_.imag))
            
            datapoint = {
                'SNR': snr,
                'X': y,
                'y': np.array([theta, r_norm])
            }
            
            dataset[ii] = datapoint
            ii += 1

    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    np.save(dataset_path,dataset,allow_pickle=True)
    print(f"Dataset saved to {dataset_path}")
    print(f'The dataset contains {len(dataset)} samples')
    exit()

# Load Data
dataset = np.load(dataset_path,allow_pickle=True).item()
print(f"Dataset {dataset_path} loaded successfully.")

train_dim = int(train_val_split*dataset_size*len(SNR))
val_dim = int(val_split*dataset_size*len(SNR))
test_dim = int(test_split*dataset_size*len(SNR))

X_train, y_train = np.array([dataset[i]['X'] for i in range(train_dim)]), np.array([dataset[i]['y'] for i in range(train_dim)])
X_val, y_val = np.array([dataset[i]['X'] for i in range(train_dim,train_dim+val_dim)]), np.array([dataset[i]['y'] for i in range(train_dim,train_dim+val_dim)])
X_test, y_test = np.array([dataset[i]['X'] for i in range(train_dim+val_dim,train_dim+val_dim + test_dim)]), np.array([dataset[i]['y'] for i in range(train_dim+val_dim,train_dim+val_dim + test_dim)])

SNR_train = torch.tensor([np.where(dataset[i]['SNR']==np.array(SNR))[0] for i in range(train_dim)]).squeeze()
SNR_val = torch.tensor([np.where(dataset[i]['SNR']==np.array(SNR))[0] for i in range(val_dim)]).squeeze()
SNR_test = torch.tensor([np.where(dataset[i]['SNR']==np.array(SNR))[0] for i in range(test_dim)]).squeeze()

print('train set: ',len(X_train))
print('val set: ',len(X_val))
print('test set: ',len(X_test))

# PyTorch Tensors
X_train = torch.tensor(X_train).float()
X_val = torch.tensor(X_val).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_val = torch.tensor(y_val).float()
y_test = torch.tensor(y_test).float()


# PyTorch GPU/CPU selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #%% Training
if args.train:
    # Model Save Folder
    model_directory = args.logdir
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    print('Saving the models to %s' % model_directory)

    # Reproducibility
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    torch.backends.cudnn.deterministic = True

    # PyTorch GPU/CPU selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Neural Network
    model = models[args.model](N,N_RF,2)
    trainable_params = list(model.parameters())
    num_trainable_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print("Number of trainable parameters:", num_trainable_params)

    # Training Settings
    model.to(device)
    criterion = torch.nn.MSELoss() # Training Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Optimizer
    if args.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, threshold=0.0001)

    train_loss = np.zeros((epochs))
    train_rmse_r = []
    train_rmse_theta = []
    train_rmse_pos = []
    val_rmse_r = []
    val_rmse_theta = []
    val_rmse_pos = []
    min_val_loss = 100
    min_pos_error = 100
    train_acc = np.zeros((epochs))
    val_loss = np.zeros((epochs))
    val_acc = np.zeros((epochs))
    best_epoch = 0

    # Epochs
    for epoch in range(epochs):
        print('Epoch %i/%i:'%(epoch+1, epochs), flush=True)
        print('lr = ',optimizer.param_groups[0]['lr'])
        
        train_loss[epoch], RMSE_train = train_loop(X_train, y_train, SNR_train, model, optimizer, criterion, device,N,N_RF,range_limits,type=args.type, batch_size=batch_size)
        val_loss[epoch], RMSE_val = eval_loop(X_val, y_val, SNR_val, model, criterion, device,N,N_RF,range_limits,type=args.type, batch_size=batch_size)
        data = {
            'SNR [dB]': SNR_dB,
            'Train (r)': RMSE_train['r'],
            'Val (r)': RMSE_val['r'],
            'Train (theta)': RMSE_train['theta'],
            'Val (theta)': RMSE_val['theta'],
            'Train (pos)': RMSE_train['pos'],
            'Val (pos)': RMSE_val['pos'],
        }
        df = pd.DataFrame(data,index=SNR_dB)
        df.index.name = 'SNR [dB]'
        print(df)
        avg_pos_error = df['Val (pos)'].mean()
        print(f'Train loss: {train_loss[epoch]:.5f}, Val loss: {val_loss[epoch]:.5f}, avg position error: {avg_pos_error:.2f} m')
        if val_loss[epoch] < min_val_loss or df['Val (pos)'].mean() < min_pos_error:
            best_epoch = epoch
            min_pos_error = df['Val (pos)'].mean()
            print(f'Val loss improved: {min_val_loss:.5f} -> {val_loss[epoch]:.5f}')
            print('Saving results...')
            min_val_loss = val_loss[epoch]
            print('Saving model..')
            torch.save(model.state_dict(), os.path.join(model_directory, 'model_best.pth'))
            df.to_csv(os.path.join(args.logdir,'best_rmse_epoch.csv'),index=False)
        print(f'Best epoch: {best_epoch+1}, best val loss: {min_val_loss:.5f}, best (avg) position error: {min_pos_error:2f}\n')

        train_rmse_r.append(RMSE_train['r'])
        train_rmse_theta.append(RMSE_train['theta'])
        train_rmse_pos.append(RMSE_train['pos'])
        val_rmse_r.append(RMSE_val['r'])
        val_rmse_theta.append(RMSE_val['theta'])
        val_rmse_pos.append(RMSE_val['pos'])

    torch.save(model.state_dict(), os.path.join(model_directory, 'model_final.pth'))
    df.to_csv(os.path.join(args.logdir,'final_rmse_epoch.csv'),index=False)
    np.save(os.path.join(model_directory, 'train_loss.npy'),train_loss)
    np.save(os.path.join(model_directory, 'val_loss.npy'),val_loss)
    np.save(os.path.join(model_directory, 'train_rmse_r.npy'),train_rmse_r)
    np.save(os.path.join(model_directory, 'train_rmse_theta.npy'),train_rmse_theta)
    np.save(os.path.join(model_directory, 'train_rmse_pos.npy'),train_rmse_pos)
    np.save(os.path.join(model_directory, 'val_rmse_r.npy'),val_rmse_r)
    np.save(os.path.join(model_directory, 'val_rmse_theta.npy'),val_rmse_theta)
    np.save(os.path.join(model_directory, 'val_rmse_pos.npy'),val_rmse_pos)
    
    plt.figure()
    plt.plot(train_loss,'k',label='train loss')
    plt.plot(val_loss,'r',label='val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    
    plt.figure()
    plt.plot(train_rmse_r,'k',label='train rmse (r)')
    plt.plot(val_rmse_r,'r',label='val rmse (r)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (m)')
    
    plt.figure()
    plt.plot(train_rmse_theta,'k',label='train rmse (theta)')
    plt.plot(val_rmse_theta,'r',label='val rmse (theta)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (deg)')
    
    plt.figure()
    plt.plot(train_rmse_pos,'k',label='train rmse (pos)')
    plt.plot(val_rmse_pos,'r',label='val rmse (pos)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (m)')
    plt.show()

    print('Finished Training')

    
    print('Testing..')
    model_PATH = args.logdir + '/model_best.pth'
    model = models[args.model](N,N_RF,2)
    model.load_state_dict(torch.load(model_PATH))
    
    criterion = torch.nn.MSELoss() # Training Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Optimizer

    test_loss, RMSE_test, data_test = test_loop(X_test, y_test, SNR_test, model, criterion, device,N,N_RF,range_limits, batch_size=args.batch_size,type=args.type)
    data = {
            'Test (r)': RMSE_test['r'],
            'Test (theta)': RMSE_test['theta'],
            'Test (pos)': RMSE_test['pos']
        }
    df = pd.DataFrame(data,index=SNR_dB)
    df.index.name = 'SNR [dB]'
    print(df)
    df.to_csv(os.path.join(args.logdir,'test_rmse.csv'),index=False)
    df_datapoints = pd.DataFrame(data_test)
    df_datapoints.to_csv(os.path.join(args.logdir,'test_scores.csv'),index=False)
    
else:
    #%% Test

    print('Testing..')
    model_PATH = args.logdir + '/model_best.pth'
    model = models[args.model](N,N_RF,2)
    model.load_state_dict(torch.load(model_PATH))
    
    criterion = torch.nn.MSELoss() # Training Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Optimizer

    test_loss, RMSE_test, data_test = test_loop(X_test, y_test, SNR_test, model, criterion, device,N,N_RF,range_limits, batch_size=args.batch_size,type=args.type)
    data = {
            'Test (r)': RMSE_test['r'],
            'Test (theta)': RMSE_test['theta'],
            'Test (pos)': RMSE_test['pos']
        }
    df = pd.DataFrame(data,index=SNR_dB)
    df.index.name = 'SNR [dB]'
    print(df)
    df.to_csv(os.path.join(args.logdir,'test_rmse.csv'),index=False)
    df_datapoints = pd.DataFrame(data_test)
    df_datapoints.to_csv(os.path.join(args.logdir,'test_scores.csv'),index=False)