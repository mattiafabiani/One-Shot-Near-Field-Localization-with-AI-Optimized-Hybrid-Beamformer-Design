import argparse
import json
import os, sys
import csv

from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import torch.nn as nn
torch.backends.cudnn.benchmark = True
# from scheduler import CyclicCosineDecayLR

from data2_seq import MATTIA_Data
from config_seq import GlobalConfig
from all_models import LSTM_GPS, LateFusionLSTM
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torchvision

kw='best_'# keyword for the pretrained model in finetune

torch.cuda.empty_cache()
# torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,                   default='', help='Unique experiment identifier')
parser.add_argument('--device', type=str,               default='cuda:0', help='Device to use')
parser.add_argument('--model', type=int,                default=0, help='Model to use')
parser.add_argument('--epochs', type=int,               default=80, help='Number of train epochs.')
parser.add_argument('--clean_noisy_labels', type=int,   default=0, help='Clean noisy labels (best beam not existing)')
parser.add_argument('--drop', type=float,               default=0., help='Learning rate.')
parser.add_argument('--lr', type=float,                 default=1e-2, help='Learning rate.')
parser.add_argument('--batch_size', type=int,           default=64, help='Batch size')
parser.add_argument('--train_split', type=float,        default=0.8, help='Train split')
parser.add_argument('--show_distribution', type=int,    default=0, help='shows distribution of train/val sets')
parser.add_argument('--seconds_per_chunk', type=int,    default=3, help='How many seconds for each validation separate group')
parser.add_argument('--logdir', type=str,               default='log', help='Directory to log data to')
parser.add_argument('--gps_features', type = int,       default=0, help='use more normalized GPS features')
parser.add_argument('--scheduler', type=int,            default=1, help='use scheduler to control the learning rate')
parser.add_argument('--load_previous_best', type=int,   default=0, help='load previous best pretrained model ')
parser.add_argument('--Test', type=int,                 default=0, help='Test')
parser.add_argument('--ema', type=int,                  default=1, help='exponential moving average')
args = parser.parse_args()
foldername = str(args.epochs)+'epochs_'+str(args.batch_size)+'batch_'+'model'+str(args.model)+'_drop'+str(args.drop)+'_lr'+str(args.lr)+'_'
model_type = ['gps', 'gps_plus_tracking', 'tracking', 'gps_plus_tracking_condition']
args.logdir = os.path.join(args.logdir, foldername + args.id + model_type[args.model])

writer = SummaryWriter(log_dir=args.logdir)

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class Engine(object):
	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.APL = [-100]
		self.bestval = -100
		self.APLft = [-100]
		self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()
		pred_beam_all = []
		gt_pwr_all = []
		pred_pwr_all = []
		all_true_beams = []
		# Train loop
		pbar=tqdm(dataloader_train, desc='training')
		for data in pbar:
			# efficiently zero gradients
			optimizer.zero_grad(set_to_none=True)
			# image_front_list = []
			# image_back_list = []
			# for i in range(config.seq_len):
			# 	image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32))
			# 	image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))

			x = data[model_type[args.model]].to(args.device, dtype=torch.float32)
			# exit()
			# pred_beams = model(image_front_list,image_back_list,x)
			# print(pred_beams.shape)
			# exit()
			pred_beams = model(x)
			# pred_beam = torch.remainder( torch.argmax(pred_beams, dim=1)+1, 256 ) # predicted index (+1 shift)
			pred_beam = torch.argmax(pred_beams, dim=1)
			gt_beamidx = data['beamidx'].to(args.device, dtype=torch.long)
			gt_beams = data['beam'].to(args.device, dtype=torch.float32)
			# running_acc = (pred_beam == gt_beamidx).sum().item()
			loss = self.criterion(pred_beams, gt_beams)
			# gt_beam_all.append(data['beamidx'][0])
			# pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
			all_true_beams.append(data['all_true_beams']) #.cpu().numpy()
			pred_beam_all.append(pred_beam.cpu().numpy())
			true_pwr_batch = data['beam_pwr'].to(args.device, dtype=torch.float32)
			gt_beamidx_shifted = torch.remainder( gt_beamidx + 1, 256)
			# gt_beamidx_shifted = gt_beamidx
			gt_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx_shifted]).cpu().numpy())
			pred_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy())

			loss.backward()
			loss_epoch += float(loss.item())
			pbar.set_description(str(loss.item()))
			num_batches += 1
			optimizer.step()
			if args.ema:# Exponential Moving Averages
				ema.update()	# during training, after update parameters, update shadow weights

			self.cur_iter += 1
		pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
		all_true_beams = np.squeeze(np.concatenate(all_true_beams, 0))
		pred_pwr_all = np.squeeze(np.concatenate(pred_pwr_all, 0))
		gt_pwr_all = np.squeeze(np.concatenate(gt_pwr_all, 0))
		curr_acc = compute_acc(all_true_beams, pred_beam_all)
		APL_score = APL(gt_pwr_all, pred_pwr_all)
		print('Train top beam acc: ',curr_acc, ' APL score: ',APL_score)
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		# self.cur_epoch += 1
		writer.add_scalar('APL_score_train', APL_score, self.cur_epoch)
		for i in range(len(curr_acc)):
			writer.add_scalars('curr_acc_train', {'beam' + str(i):curr_acc[i]}, self.cur_epoch)
		writer.add_scalar('curr_loss_train', loss_epoch, self.cur_epoch)
		# if args.finetune:
		if APL_score > self.APLft[-1]:
			self.APLft.append(APL_score)
			print(f'train APL: {self.APLft[-2]:.3f} dB -> {APL_score:.3f} dB')
		else:
			print('best APL: ',self.APLft[-1], ' dB')
		return loss_epoch, APL_score

	def validate(self):
		if args.ema:#Exponential Moving Averages
			ema.apply_shadow()    # before eval\uff0capply shadow weights
		model.eval()
		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.
			gt_beam_all=[]
			pred_beam_all=[]
			all_true_beams = []
			scenario_all = []
			gt_pwr_all = [] # added co compute APL score
			pred_pwr_all = []
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val),0):
				# create batch and move to GPU
				# image_front_list = []
				# image_back_list = []
				# for i in range(config.seq_len):
				# 	image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32))
				# 	image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))

				x = data[model_type[args.model]].to(args.device, dtype=torch.float32)
				# exit()
				# pred_beams = model(image_front_list,image_back_list,x)
				x = data[model_type[args.model]].to(args.device, dtype=torch.float32)
				pred_beams = model(x)
				# pred_beam = torch.remainder( torch.argmax(pred_beams, dim=1)+1, 256 ) # predicted index (+1 shift)
				pred_beam = torch.argmax(pred_beams, dim=1)
				gt_beamidx = data['beamidx'].to(args.device, dtype=torch.long)
				gt_beams = data['beam'].to(args.device, dtype=torch.float32)
				# running_acc = (pred_beam == gt_beamidx).sum().item()
				loss = self.criterion(pred_beams, gt_beams)
				# gt_beam_all.append(data['beamidx'][0])
				# pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
				all_true_beams.append(data['all_true_beams']) #.cpu().numpy()
				pred_beam_all.append(pred_beam.cpu().numpy())
				true_pwr_batch = data['beam_pwr'].to(args.device, dtype=torch.float32)
				gt_beamidx_shifted = torch.remainder( gt_beamidx + 1, 256)
				# gt_beamidx_shifted = gt_beamidx
				gt_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx_shifted]).cpu().numpy())
				pred_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy())
				scenario_all.append(data['scenario'])
				num_batches += 1
				wp_epoch += loss.cpu().numpy()
    
			all_true_beams = np.concatenate(all_true_beams,0) # (batch,256)
			pred_beam_all = np.concatenate(pred_beam_all,0) # (batch,)
			# gt_beam_all=np.squeeze(np.concatenate(gt_beam_all,0))
			scenario_all = np.squeeze(np.concatenate(scenario_all,0))
			pred_pwr_all = np.squeeze(np.concatenate(pred_pwr_all, 0)) # (n_samples,1)
			gt_pwr_all = np.squeeze(np.concatenate(gt_pwr_all, 0)) # (n_samples,1)
			#calculate accuracy and APL score according to different scenarios
			# scenarios = ['scenario36', 'scenario37', 'scenario38', 'scenario39']
			scenarios = ['scenario36']
			for s in scenarios:
				beam_scenario_index = np.array(scenario_all) == s
				pred_pwr_s = pred_pwr_all[beam_scenario_index]
				gt_pwr_s = gt_pwr_all[beam_scenario_index]
				if np.sum(beam_scenario_index) > 0:
					curr_acc_s = compute_acc(all_true_beams[beam_scenario_index],pred_beam_all[beam_scenario_index])
					APL_score_s = APL(gt_pwr_s,pred_pwr_s)
					
					print(s, ' curr_acc: ', curr_acc_s, ' APL_score: ', APL_score_s)
					for i in range(len(curr_acc_s)):
						writer.add_scalars('curr_acc_val', {s + 'beam' + str(i):curr_acc_s[i]}, self.cur_epoch)
					writer.add_scalars('APL_score_val', {s:APL_score_s}, self.cur_epoch)

			curr_acc = compute_acc(all_true_beams, pred_beam_all)
			APL_score_val = APL(gt_pwr_all, pred_pwr_all)
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:d}, Batch {batch_num:d}:' + f' Loss: {wp_loss:3.3f}')
			print('Val top beam acc: ',curr_acc, 'APL score: ', APL_score_val)
			print(f'Losing {(1 - pow(10,APL_score_val/10))*100:.2f} % of the power')
			writer.add_scalars('APL_score_val', {'scenario_all':APL_score_val}, self.cur_epoch)
			writer.add_scalar('curr_loss_val', wp_loss, self.cur_epoch)

			self.val_loss.append(wp_loss)
			self.APL.append(float(APL_score_val))
			self.cur_epoch += 1

		if args.ema: # Exponential Moving Averages
			ema.restore()	# after eval, restore model parameter
		return wp_loss, APL_score_val


	def test(self):
		model.eval()
		with torch.no_grad():
			pred_beam_all=[]
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_test), 0): # CHANGE TO dataloader_test
				# create batch and move to GPU
				image_front_list = []
				image_back_list = []
				gps_list = []
				# for i in range(config.seq_len):
				# 	image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32))
				# 	image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
    
				for i in range(config.n_gps):
					gps_list.append(data['gps'][i].to(args.device, dtype=torch.float32))
      
				pred_beams = model(gps_list)
				pred_beam = torch.argmax(pred_beams, dim=1)
				pred_beam_all.append(pred_beam.cpu().numpy())

			pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
			df_out = pd.DataFrame()
			df_out['prediction'] = pred_beam_all
			df_out.to_csv('beamwise_prediction_custom_test_set.csv', index=False)

	def save(self):
		save_best = False

		if self.APL[-1] >= self.bestval:
			self.bestval = self.APL[-1]
			self.bestval_epoch = self.cur_epoch - 1
			save_best = True
		print(f'best APL = {self.bestval:.4f} dB @ epoch = {self.bestval_epoch}')

		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
			'APL': self.APL,
		}

		# Save ckpt for every epoch
		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'final_model.pth'))
		# # Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))


		if save_best:# save the bestpretrained model
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model_val.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim_val.pth'))
			tqdm.write('====== Overwrote best model ======>')
		elif args.load_previous_best:
			model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model_val.pth')))
			optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'best_optim_val.pth')))
			tqdm.write('====== Load the previous best model ======>')

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def compute_acc(all_beams, only_best_beam, top_k=[1, 3, 5]):
    
    """ 
    Computes top-k accuracy given prediction and ground truth labels.

    Note that it works bidirectionally. 
    <all_beams> is (N_SAMPLES, N_BEAMS) but it can represent:
        a) the ground truth beams sorted by receive power
        b) the predicted beams sorted by algorithm's confidence of being the best

    <only_best_beam> is (N_SAMPLES, 1) and can represent (RESPECTIVELY!):
        a) the predicted optimal beam index
        b) the ground truth optimal beam index

    For the competition, we will be using the function with inputs described in (a).

    """
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)

    n_test_samples = len(only_best_beam)
    if len(all_beams) != n_test_samples:
        raise Exception(
            'Number of predicted beams does not match number of labels.')

    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(only_best_beam)):
        for k_idx in range(n_top_k):
            hit = np.any(all_beams[samp_idx, :top_k[k_idx]] == only_best_beam[samp_idx])
            total_hits[k_idx] += 1 if hit else 0

    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(only_best_beam)*100, 4)


def APL(true_best_pwr, est_best_pwr):
    """
    Average Power Loss: average of the power wasted by using the predicted beam
    instead of the ground truth optimum beam.
    """
    return np.mean(10 * np.log10(est_best_pwr / true_best_pwr))


# Config
config = GlobalConfig()
config.gps_features = args.gps_features
config.n_gps_features_max = 10

import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed) # numpy
torch.manual_seed(seed) # torch+CPU
torch.cuda.manual_seed(seed) # torch+GPU
torch.use_deterministic_algorithms(False)
g = torch.Generator()
g.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============ DATASET

# train_root = config.root + '/Multi_Modal/scenario36/'
train_root = config.root
# train_root_csv = 'dataset_rnorm60_utm.csv'
# train_root_csv = 'dataset_rnorm_utm.csv'
train_root_csv = 'dataset_tracked.csv'

# custom train/val/test split (non-overlapping samples before shuffling)
df = pd.read_csv('TX_Tracking_Problem/LSTM_datasets/' + train_root_csv)
SAMPLING_FREQ = 10
# idx = np.where(df['avg_detections'].values == np.max(df['avg_detections'].values))
# print(df['x1_unique_index'].iloc[idx[0]])
# exit()

# divide the dataset into chunks of args.seconds_per_chunk seconds
grouped_df = df.groupby(df.index // (args.seconds_per_chunk*SAMPLING_FREQ))
# concatenate each group into a list (each element is a DataFrame with 200 sequences -> 20s)
grouped_list = [group for _, group in grouped_df]
train_groups, val_groups = train_test_split(grouped_list, test_size=1 - args.train_split, random_state=seed)

# remove overlapping sequences between train and validation sets
for i, val_group in enumerate(val_groups):
    val_group.drop(val_group.index[:5], inplace=True) # separates train/val sequences
    val_group.drop(val_group.index[-5:], inplace=True)
train_df = pd.concat(train_groups, ignore_index=True)
val_df = pd.concat(val_groups, ignore_index=True)

if args.clean_noisy_labels:
	train_set_len = len(train_df)
	count = 0
	i = 0
	PAPR_THR = 8
	print('=======removing noisy labels (PAPR < 1.3)')
	progress_bar = tqdm(total=train_set_len)
	while True:
		# if i % 1000 == 0:
			# print(f'iteration {i}')
		y_pwrs = np.array(train_df['y1_pwr_vec'][i].strip('[]').split()).astype(float)
		# y_pwrs = np.zeros((4,64))
		# for arr_idx in range(4): # 4 antenna arrays
		# 	y_pwrs[arr_idx,:] = np.loadtxt(train_root + train_df[f'y1_unit1_pwr{arr_idx+1}'].iloc[i])
		# y_pwrs = y_pwrs.reshape((256)) # N_ARR*N_BEAMS
		PAPR = y_pwrs.max() / y_pwrs.mean()
		if PAPR <= PAPR_THR:
			count += 1
			# train_df.drop(train_df.index[i],inplace=False)
			train_df.drop(index=i,inplace=False)
		progress_bar.update(1)
		i += 1
		if i == len(train_df):
			break
	progress_bar.close()
	print('=======finished\n')
	print(f'Train set: {train_set_len} to {len(train_df)}')
	print(f'# of removed sequences: {count} ({count/train_set_len*100:.3f} %)')
# print(train_df.iloc[5442])
# exit()

train_set = MATTIA_Data(root=train_root, dataframe=train_df, config=config, test=False)
val_set = MATTIA_Data(root=train_root, dataframe=val_df, config=config, test=False)
# test_set = MATTIA_Data(root=train_root, dataframe=df_test, config=config, test=True)
print('train_set:', len(train_set))
print('val_set:', len(val_set))

if args.show_distribution:
    '''
    PLOT DISTRIBUTIONS OF THE TRAIN AND VAL SETS
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(val_df['y1_gt_beam'], bins=256, alpha=0.7,label='val')
    ax.hist(train_df['y1_gt_beam'], bins=256, alpha=0.4, label='train')
    # ax.set_title(f'Val Set Distribution')
    ax.set_xlabel('Beam index')
    ax.set_ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(val_df['scenario'], alpha=0.7)
    ax.set_title(f'Val Set Distribution')
    ax.set_xlabel('Beam index')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
# val_set, test_set = torch.utils.data.random_split(val_set,[len(val_set) - 500, 500])


if args.Test:
    # custom test set
	# dataloader_test = DataLoader(test_set,batch_size=args.batch_size,shuffle=False, pin_memory=False, generator=g)
	# competition test set
	test_root = config.root + '/Multi_Modal_Test/'
	test_root_csv = 'challenge.csv'
	test_set = MATTIA_Data(root=test_root, root_csv=test_root_csv, config=config, test=True)
	print('test_set:', len(test_set))
	dataloader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=False)

## random train/val split (wrong for time series)
if not args.Test:						 
	dataloader_train = DataLoader(train_set,batch_size=args.batch_size,shuffle=True, pin_memory=True,
                                    worker_init_fn=seed_worker, generator=g)
	dataloader_val = DataLoader(val_set,batch_size=args.batch_size,shuffle=False, pin_memory=False,
                                worker_init_fn=seed_worker, generator=g)
	# dataloader_test = DataLoader(val_set,batch_size=args.batch_size,shuffle=False, pin_memory=False, generator=g)
else:
	test_root = config.root + '/Multi_Modal_Test/'
	test_root_csv = 'challenge.csv'
	test_set = MATTIA_Data(root=test_root, root_csv=test_root_csv, config=config, test=True)
	print('test_set:', len(test_set))
	dataloader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=False,collate_fn=lambda x: x)

# print(next(iter(dataloader_train)))
# exit()
# Model
input_size = next(iter(dataloader_train))[model_type[args.model]].shape[2]
# print(next(iter(dataloader_train))[model_type[args.model]].shape)
# exit()
n_hidden = 128
n_out = 256
n_stacked_layers = 2
model = LSTM_GPS(input_size,n_hidden,n_out,n_stacked_layers,args.drop).to(device=args.device)
# model = LateFusionLSTM().to(args.device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)#,weight_decay=1e-5)
# early_stopping = EarlyStopping(tolerance=5,min_delta=1)
if args.scheduler:#Cyclic Cosine Decay Learning Rate
	scheduler = ReduceLROnPlateau(optimizer, 'min')
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
	# scheduler = CyclicCosineDecayLR(optimizer, 
	#                                 init_decay_epochs=7, # 15
	#                                 min_decay_lr=1e-4, # 2.5e-6
	#                                 restart_interval = 5, # 10
	#                                 restart_lr= 1e-2, # 12.5e-5
	#                                 warmup_epochs=5, # 10
	#                                 warmup_start_lr=0.001) # 2.5e-6
 
trainer = Engine()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('======Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('======Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('======Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']
	trainer.APL = log_table['APL']

	print('======loading best_model')
	model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model_val.pth')))
	optimizer.load_state_dict(torch.load(os.path.join(args.logdir,'best_optim_val.pth')))


ema = EMA(model, 0.999)

if args.ema:
	ema.register()

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)



if args.Test:
	trainer.test()
	print('Test finish')
else:
    
	# val_loss, val_APL = trainer.validate()
	# exit()
	for epoch in range(trainer.cur_epoch, args.epochs):
		print('\nepoch:',epoch)
		print('lr = ',optimizer.param_groups[0]['lr'])
		writer.add_scalar('lr', optimizer.param_groups[0]['lr'], trainer.cur_epoch)
		train_loss, train_APL = trainer.train()
		val_loss, val_APL = trainer.validate()
		writer.add_scalars('train_val_loss', {'train_loss': train_loss, 'val_loss': val_loss}, trainer.cur_epoch)
		writer.add_scalars('train_val_APL', {'train_APL': train_APL, 'val_APL': val_APL}, trainer.cur_epoch)
		# early_stopping(train_loss,val_loss)
		trainer.save()

		if args.scheduler:
			scheduler.step(val_loss)

