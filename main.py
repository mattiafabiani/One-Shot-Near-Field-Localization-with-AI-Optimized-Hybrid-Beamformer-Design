import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from dnn_model import DNN_model
from network_functions import train_loop, eval_loop, test_loop, evaluate_predictions


#---------- SIMULATION PARAMETERS -----------
f0 = 25e9                   # carrier frequency
k = 2*np.pi / (3e8 / f0)    # wave number
d = 3e8/f0 / 2              # antenna spacing
N = 128                     # antennas
N_RF = 16                   # RF chains
SNR_dB = list(range(0,25,5))
SNR = [10 ** (SNR / 10) for SNR in SNR_dB]

range_limits = [1, 20]      # near-field range limits [m]

dataset_size = 10000        # number of signals per SNR (10k * 5 = 50k samples in ottal)
generate_dataset = False
train = True
epochs = 50
batch_size = 64
lr = 1e-2
train_val_split = 0.8
val_split = 0.1
test_split = 0.1
dataset_root = 'dataset/'
dataset_path = os.path.join(dataset_root,'dataset_dummy.npy')
models = [DNN_model]        # create a list of models if more than one are developed
rng_seed = 42
#------------------------------------------

#%% Data Prep

if generate_dataset:
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

print('train set: ',len(X_train))
print('val set: ',len(X_val))
print('test set: ',len(X_test))

# PyTorch Tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_val = torch.from_numpy(y_val).float()
y_test = torch.from_numpy(y_test).float()

#%% Training
data_type = 0
# Model Save Folder
folder_name = 'type%i_batchsize%i_rng%i_epoch%i' %(data_type, batch_size, rng_seed, epochs)
models_directory = os.path.abspath('./saved_models/')
models_directory = 'saved_models/'
if not os.path.exists(models_directory):
    os.makedirs(models_directory)
c = 0
while os.path.exists(os.path.join(models_directory, folder_name + '_v%i'%c, '')):
    c += 1
model_directory = os.path.join(models_directory, folder_name + '_v%i'%c, '')
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
print('Saving the models to %s' % models_directory)

# Reproducibility
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)
torch.backends.cudnn.deterministic = True

# PyTorch GPU/CPU selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class weightConstraint(object):
#     def __init__(self,N,N_RF):
#         self.N = N
#         self.N_RF = N_RF
#         pass
    
#     def __call__(self,module):
#         if hasattr(module,'weight'):
#             w = module.weight.data # shape (2*N_RF,2*N)
#             w_rf = w[:self.N_RF,:self.N] + 1j*w[self.N_RF:self.N_RF*2,:self.N] # shape (N_RF,N)
#             w[:self.N_RF,:self.N] = w[:self.N_RF,:self.N] / (np.abs(w_rf) * np.sqrt(self.N))
#             w[self.N_RF:self.N_RF*2,:self.N] = w[self.N_RF:self.N_RF*2,:self.N] / (np.abs(w_rf) * np.sqrt(self.N))
#             module.weight.data=w
            
# Neural Network
model = models[data_type](N,N_RF)

# Applying the constraints to the first layer only (precoder power constraint)
# constraints=weightConstraint(N,N_RF)
# model._modules['fc1'].apply(constraints)
trainable_params = list(model.parameters())
num_trainable_params = sum(p.numel() for p in trainable_params if p.requires_grad)
print("Number of trainable parameters:", num_trainable_params)

if train:
    # Training Settings
    model.to(device)
    criterion = torch.nn.MSELoss() # Training Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loss = np.zeros((epochs))
    train_rmse_r = np.zeros((epochs))
    train_rmse_theta = np.zeros((epochs))
    train_rmse_pos = np.zeros((epochs))
    val_rmse_r = np.zeros((epochs))
    val_rmse_theta = np.zeros((epochs))
    val_rmse_pos = np.zeros((epochs))
    train_acc = np.zeros((epochs))
    val_loss = np.zeros((epochs))
    val_acc = np.zeros((epochs))

    # Epochs
    for epoch in range(epochs):
        print('Epoch %i/%i:'%(epoch+1, epochs), flush=True)
        
        train_loss[epoch], train_rmse_r[epoch], train_rmse_theta[epoch], train_rmse_pos[epoch] = train_loop(X_train, y_train, model, optimizer, criterion, device,N,N_RF,range_limits, batch_size=batch_size)
        val_loss[epoch], val_rmse_r[epoch], val_rmse_theta[epoch], val_rmse_pos[epoch] = eval_loop(X_val, y_val, model, criterion, device,N,N_RF,range_limits, batch_size=batch_size)
        
        # Save the best model
        if val_loss[epoch] <= np.min(val_loss[:epoch] if epoch>0 else val_loss[epoch]):
            print('Saving model..')
            torch.save(model.state_dict(), os.path.join(model_directory, 'model_best.pth'))
            # torch.save(model.state_dict(), 'saved_models/type0_batchsize5_rng42_epoch500_v11/model_best.pth')
            
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(model_directory, 'model_final.pth'))
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
else:
    #%% Test

    print('Testing..')
    # topk=5
    model_directory = 'saved_models/type0_batchsize64_rng42_epoch50_v0'
    network_time_per_sample = test_loop(X_test, y_test, model, device,N,N_RF,range_limits, model_path=os.path.join(model_directory, 'model_best.pth')) # Best model
    # topk_acc_best, beam_dist_best = evaluate_predictions(y, y_hat, k=topk)
    # print('Best model:')
    # print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_best*100))
    # print('Beam distance: %.2f' % beam_dist_best)


    # y_final, y_hat_final, network_time_per_sample_final = test_loop(X_test, y_test, model, device, model_path=os.path.join(model_directory, 'model_final.pth')) # Last Epoch
    # topk_acc_final, beam_dist_final = evaluate_predictions(y_final, y_hat_final, k=topk)
    # print('Final model:')
    # print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_final*100))
    # print('Beam distance: %.2f' % beam_dist_final)