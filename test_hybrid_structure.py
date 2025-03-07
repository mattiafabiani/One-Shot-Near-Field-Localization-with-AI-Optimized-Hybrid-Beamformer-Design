from dnn_model import *
import numpy as np

N = 8
N_RF = 4

net = DNN_model(N=N,N_RF=N_RF)


for name, params in net.named_parameters():
    if 'weight' in name and 'fc1' in name:
        print('Weights fc1 BEFORE weights update')
        print(params.detach())
        # w = params[:N_RF/2,:N] + 1j*params[N_RF/2:N_RF,:N]
        # sum_weights = np.sum(np.abs(w.detach().numpy())**2)
        # print(sum_weights)

# constraints = weightConstraint(N,N_RF,'sub-connected')
constraints = weightConstraint(N,N_RF,'sub-connected')
net._modules['fc1'].apply(constraints)


for name, params in net.named_parameters():
    if 'weight' in name and 'fc1' in name:
        print('\nWeights fc1 AFTER weights update')
        print(params.detach())
        w = params[:N_RF,:N] + 1j*params[N_RF:2*N_RF,N:2*N]
        # w = params[0,0] + 1j*params[2,4]
        abs_w = np.abs(w.detach().numpy())**2
        print(abs_w)
        print(np.sum(abs_w,axis=1))