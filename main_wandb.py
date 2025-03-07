'''
Author: Mattia Fabiani
Update: This version of the main allows the grouping of the results by SNR values.
'''

import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import torch
import argparse
import wandb
import torch.optim as optim
import os
# from torch.utils.data import DataLoader
from dnn_model import DNN_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from network_functions import train_loop, eval_loop, test_loop, evaluate_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,                   default='', help='Unique experiment identifier')
parser.add_argument('--epochs', type=int,               default=50, help='Number of train epochs.')
parser.add_argument('--N', type=int,                    default=128, help='Number of antennas.')
parser.add_argument('--N_RF', type=int,                 default=16, help='Number of RF chains.')
parser.add_argument('--type', type=str,                 default='fully-connected', help='RF chains to antenna connection')
parser.add_argument('--generate_dataset', type=int,     default=0, help='Generate Dataset.')
parser.add_argument('--train', type=int,                default=1, help='Train the DNN model.')
# parser.add_argument('--drop', type=float,               default=0., help='Dropout.')
parser.add_argument('--lr', type=float,                 default=1e-3, help='Learning rate.')
parser.add_argument('--batch_size', type=int,           default=250, help='Batch size')
parser.add_argument('--train_split', type=float,        default=0.8, help='Train split')
parser.add_argument('--logdir', type=str,               default='saved_models', help='Directory to log data to')
parser.add_argument('--scheduler', type=int,            default=0, help='use scheduler to control the learning rate')
args = parser.parse_args()
foldername = args.type + '_' +'epochs'+str(args.epochs)+'_batch'+str(args.batch_size)+'_lr'+str(args.lr)+'_'+str(args.N_RF)+'RF_'+str(args.N)+'N'
args.logdir = os.path.join(args.logdir, foldername + '_' + args.id)

#---------- SIMULATION PARAMETERS -----------
f0 = 25e9                   # carrier frequency
k = 2*np.pi / (3e8 / f0)    # wave number
d = 3e8/f0 / 2              # antenna spacing
N = args.N                     # antennas
N_RF = args.N_RF                   # RF chains
SNR_dB = list(range(0,25,5))
SNR = [10 ** (SNR / 10) for SNR in SNR_dB]

range_limits = [1, 20]      # near-field range limits [m]

dataset_size = 20000        # number of signals per SNR (10k * 5 = 50k samples in ottal)
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
train_val_split = args.train_split
val_split = 0.1
test_split = 0.1
dataset_root = 'dataset/'
dataset_path = os.path.join(dataset_root,'dataset.npy')
models = [DNN_model]        # create a list of models if more than one are developed
rng_seed = 42
#------------------------------------------

wandb.init(
    project="DNN Localization",  # Set your project name here
    name='first_run',
    config={
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
    }   # Pass all the arguments to the config section of WandB
)
sweep_config = {
    'method': 'random'
    }
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'fc_layer_size': {
        'values': [128, 256, 512]
        },
    'dropout': {
          'values': [0.3, 0.4, 0.5]
        },
    }

sweep_config['parameters'] = parameters_dict
sweep_config['metric'] = metric
config = wandb.config

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
    for i in range(dataset_size):
        for snr in SNR:
            sigma_n = 1 / np.sqrt(snr)
            s = CN_realization(mean=0, std_dev=1)
            n = CN_realization(mean=0, std_dev=sigma_n, size=N)
            
            r = np.random.uniform(range_limits[0], range_limits[1])
            r_norm = 2 * (r - range_limits[0]) / (range_limits[1] - range_limits[0]) - 1
            theta = np.random.uniform(-1,1)
            
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
    print(len(dataset))

# Load Data
dataset = np.load(dataset_path,allow_pickle=True).item()
print("Dataset loaded successfully.")

train_dim = int(train_val_split*dataset_size*len(SNR))
val_dim = int(val_split*dataset_size*len(SNR))
test_dim = int(test_split*dataset_size*len(SNR))

X_train, y_train = np.array([dataset[i]['X'] for i in range(train_dim)]), np.array([dataset[i]['y'] for i in range(train_dim)])
X_val, y_val = np.array([dataset[i]['X'] for i in range(train_dim,train_dim+val_dim)]), np.array([dataset[i]['y'] for i in range(train_dim,train_dim+val_dim)])
X_test, y_test = np.array([dataset[i]['X'] for i in range(train_dim+val_dim,train_dim+val_dim + test_dim)]), np.array([dataset[i]['y'] for i in range(train_dim+val_dim,train_dim+val_dim + test_dim)])
# X_test, y_test = np.array([dataset[i] for i in range(train_dim+val_dim,train_dim+val_dim + test_dim)]), np.array([dataset[i]['y'] for i in range(train_dim+val_dim,train_dim+val_dim + test_dim)])
SNR_train = torch.tensor([np.where(dataset[i]['SNR']==np.array(SNR))[0] for i in range(train_dim)]).squeeze()
SNR_val = torch.tensor([np.where(dataset[i]['SNR']==np.array(SNR))[0] for i in range(val_dim)]).squeeze()
SNR_test = torch.tensor([np.where(dataset[i]['SNR']==np.array(SNR))[0] for i in range(test_dim)]).squeeze()

print('train set: ',len(X_train))
print('val set: ',len(X_val))
print('test set: ',len(X_test))
# X_test = np.array([dataset[i]['X'] for i in range(train_dim+val_dim,train_dim+val_dim + test_dim)])

# xy_test = [dataset[i] for i in range(train_dim+val_dim,train_dim+val_dim + test_dim)]
# df_test = pd.DataFrame([{key: value for key, value in record.items()} for record in xy_test])
# y_test = df_test[['y','SNR']].to_numpy()
# print(np.array(df_test[['X']].values).squeeze().shape)
# X_test = torch.tensor(X_test).float()
# exit()

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
    data_type = 0
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
    model = models[data_type](N,N_RF)

    # Applying the constraints to the first layer only (precoder power constraint)
    # constraints=weightConstraint(N,N_RF)
    # model._modules['fc1'].apply(constraints)
    trainable_params = list(model.parameters())
    num_trainable_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print("Number of trainable parameters:", num_trainable_params)

    # Training Settings
    model.to(device)
    criterion = torch.nn.MSELoss() # Training Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Optimizer
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if args.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, threshold=0.0001)

    train_loss = np.zeros((epochs))
    train_rmse_r = []
    train_rmse_theta = []
    train_rmse_pos = []
    val_rmse_r = []
    val_rmse_theta = []
    val_rmse_pos = []
    min_val_loss = 100
    train_acc = np.zeros((epochs))
    val_loss = np.zeros((epochs))
    val_acc = np.zeros((epochs))

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

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss[epoch], "val_loss": val_loss[epoch]})
        wandb.watch(model)

        df = pd.DataFrame(data,index=SNR_dB)
        df.index.name = 'SNR [dB]'
        print(df)
        print(f'Train loss: {train_loss[epoch]:.4f}, Val loss: {val_loss[epoch]:.4f}')
        if val_loss[epoch] < min_val_loss:
            print('saving results...')
            print(f'Val loss improved: {min_val_loss:.3f} -> {val_loss[epoch]:.3f}')
            min_val_loss = val_loss[epoch]
            print('Saving model..\n')
            torch.save(model.state_dict(), os.path.join(model_directory, 'model_best.pth'))
            df.to_csv(os.path.join(args.logdir,'best_rmse_epoch.csv'),index=False)
            # wandb.save(os.path.join(model_directory, 'model_best.pth'))

        train_rmse_r.append(RMSE_train['r'])
        train_rmse_theta.append(RMSE_train['theta'])
        train_rmse_pos.append(RMSE_train['pos'])
        val_rmse_r.append(RMSE_val['r'])
        val_rmse_theta.append(RMSE_val['theta'])
        val_rmse_pos.append(RMSE_val['pos'])
 
        # Save the best model
        # if val_loss[epoch] <= np.min(val_loss[:epoch] if epoch>0 else val_loss[epoch]):
        #     print('Saving model..')
        #     torch.save(model.state_dict(), os.path.join(model_directory, 'model_best.pth'))
        #     # torch.save(model.state_dict(), 'saved_models/type0_batchsize5_rng42_epoch500_v11/model_best.pth')
        if args.scheduler:
            scheduler.step(val_loss[epoch])

    torch.save(model.state_dict(), os.path.join(model_directory, 'model_final.pth'))
    np.save(os.path.join(model_directory, 'train_loss.npy'),train_loss)
    np.save(os.path.join(model_directory, 'val_loss.npy'),val_loss)
    np.save(os.path.join(model_directory, 'train_rmse_r.npy'),train_rmse_r)
    np.save(os.path.join(model_directory, 'train_rmse_theta.npy'),train_rmse_theta)
    np.save(os.path.join(model_directory, 'train_rmse_pos.npy'),train_rmse_pos)
    np.save(os.path.join(model_directory, 'val_rmse_r.npy'),val_rmse_r)
    np.save(os.path.join(model_directory, 'val_rmse_theta.npy'),val_rmse_theta)
    np.save(os.path.join(model_directory, 'val_rmse_pos.npy'),val_rmse_pos)
    wandb.log({"final_train_loss": train_loss[epoch], "final_val_loss": val_loss[epoch]})
    wandb.log({"train_rmse_r": np.mean(train_rmse_r), "val_rmse_r": np.mean(val_rmse_r)})
    wandb.log({"train_rmse_theta": np.mean(train_rmse_theta), "val_rmse_theta": np.mean(val_rmse_theta)})
    
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
else:
    #%% Test

    print('Testing..')
    # topk=5
    model_directory = 'saved_models_dataset50000/type0_batchsize64_rng42_epoch50_v4_16RF'
    model_PATH = model_directory + '/model_best.pth'
    model = models[0](N,N_RF)
    model.load_state_dict(torch.load(model_PATH))
    
    criterion = torch.nn.MSELoss() # Training Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Optimizer

    val_loss, RMSE_val = eval_loop(X_val, y_val, SNR_val, model, criterion, device,N,N_RF,range_limits, batch_size=batch_size)
    
    data = {
            'Val (r)': RMSE_val['r'],
            'Val (theta)': RMSE_val['theta'],
            'Val (pos)': RMSE_val['pos']
        }
    df = pd.DataFrame(data,index=SNR_dB)
    df.index.name = 'SNR [dB]'
    print(df)
    # network_time_per_sample = test_loop(X_test, y_test, SNR_test, model, device,N,N_RF,range_limits, model_path=os.path.join(model_directory, 'model_best.pth')) # Best model
    # print(network_time_per_sample)
    
    # topk_acc_best, beam_dist_best = evaluate_predictions(y, y_hat, k=topk)
    # print('Best model:')
    # print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_best*100))
    # print('Beam distance: %.2f' % beam_dist_best)


    # y_final, y_hat_final, network_time_per_sample_final = test_loop(X_test, y_test, model, device, model_path=os.path.join(model_directory, 'model_final.pth')) # Last Epoch
    # topk_acc_final, beam_dist_final = evaluate_predictions(y_final, y_hat_final, k=topk)
    # print('Final model:')
    # print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_final*100))
    # print('Beam distance: %.2f' % beam_dist_final)