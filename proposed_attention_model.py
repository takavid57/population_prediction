# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(1)
np.random.seed(0)

class Predictor(torch.nn.Module):
	def __init__(self, num_input, num_hidden, num_output, minibatch_size, device):
		super(Predictor, self).__init__()
		self.num_hidden = num_hidden 
		self.minibatch_size = minibatch_size 
		self.num_layers = 1 

		# hidden layers
		self.rnn = torch.nn.RNN(num_input, num_hidden, num_layers = self.num_layers, nonlinearity = 'relu', dropout = 0)

		# concatenation layer
		self.linear = torch.nn.Linear(2 * num_hidden, num_output)

		self.hidden = self.init_hidden(device)

	def init_hidden(self, device):
		return (torch.zeros(self.num_layers, self.minibatch_size, self.num_hidden)).to(device)

	def forward(self, LAG, data, nine_target_index, haven_flag, device):

		self.rnn.flatten_parameters()

		# output of rnn
		hidden_output, self.hidden = self.rnn(data, self.hidden)

		surrounding_eight_indexes = np.delete(nine_target_index, 4, 1)

		i = 0 
		# context vec
		c = []
		# fill no data area with 0 
		params = []
		while i < len(data):
			if i == 0:
				if LAG == "ALL":
					ht = hidden_output[i:i+1, :, :].reshape(1, self.minibatch_size, self.num_hidden)
					c.append(ht)
				else:
					ht = hidden_output[i:i+1, :, :].reshape(1, self.minibatch_size, self.num_hidden, 1)
				
					outside_target = torch.zeros(1, 1, self.num_hidden).to(device)
		
					past_ht = np.array(torch.cat([hidden_output[i:i+1, :, :], outside_target], dim = 1).reshape(1, self.minibatch_size + 1, self.num_hidden).detach())
					past_ht = past_ht[:, surrounding_eight_indexes]

					result = np.matmul(past_ht, np.array(ht.detach()))
					param_attn = F.softmax(torch.FloatTensor(result), dim = 2)

					c.append(torch.sum(param_attn *  torch.FloatTensor(past_ht), [0, 2]).reshape(1, self.minibatch_size, self.num_hidden).to(device))

			else:
				if LAG == "ALL":
					ht = hidden_output[i:i+1, :, :].reshape(1, self.minibatch_size, self.num_hidden, 1)

					outside_target = torch.zeros(i, 1, self.num_hidden).to(device)

					past_ht = np.array(torch.cat([hidden_output[0:i, :, :], outside_target], dim = 1).reshape(i, self.minibatch_size + 1, self.num_hidden).detach())
					past_ht = past_ht[:, nine_target_index]

					if haven_flag == 1:
						haven_hidden = np.zeros((i, self.minibatch_size, 1, self.num_hidden))
						past_ht = np.concatenate([past_ht, haven_hidden], axis = 2)

				else:
					ht = hidden_output[i:i+1, :, :].reshape(1, self.minibatch_size, self.num_hidden, 1)
		
					outside_target = torch.zeros(1, 1, self.num_hidden).to(device)

					adjusted_value = 1

					if LAG == 2 and i > 1:
						adjusted_value = 2
					elif LAG == 3 and i == 2:
						adjusted_value = 2
					elif LAG == 3 and i > 2:
						adjusted_value = 3

					past_ht_surrounding = []
					for j in range(i - adjusted_value, i):
						past_ht = np.array(torch.cat([hidden_output[j:j + 1, :, :], outside_target], dim = 1).reshape(1, self.minibatch_size + 1, self.num_hidden).detach())
						past_ht = past_ht[:, surrounding_eight_indexes]

						past_ht_surrounding.append(past_ht)

					past_ht = np.concatenate(past_ht_surrounding, axis = 2)

					past_ht_surrounding.clear()	

					past_hidden = hidden_output[0:i, :, :].permute(1, 0, 2).reshape(1, self.minibatch_size, i, self.num_hidden)

					past_ht = np.concatenate([past_ht, np.array(past_hidden.detach())], axis = 2)

					if haven_flag == 1:
						haven_hidden = np.zeros((1, self.minibatch_size, 1, self.num_hidden))
						past_ht = np.concatenate([past_ht, haven_hidden], axis = 2)

				result = np.matmul(past_ht, np.array(ht.detach()))
				param_attn = F.softmax(torch.FloatTensor(result), dim = 2)

				params.append(param_attn)	

				c.append(torch.sum(param_attn *  torch.FloatTensor(past_ht), [0, 2]).reshape(1, self.minibatch_size, self.num_hidden).to(device))

			i += 1

		concatenated_output = torch.cat([torch.cat(c, dim = 0), hidden_output], dim = 2)

		result = self.linear(concatenated_output)
		prediction_value = F.relu(result)

		return prediction_value, params

class EarlyStopping():
	def __init__(self, patience=0, verbose=0):
		self._step = 0
		self._loss = float('inf')
		self.patience = patience
		self.verbose = verbose

	def validate(self, loss):
		if self._loss < loss:
			self._step += 1
			if self._step > self.patience:
				if self.verbose:
					print('early stopping')
				return True
		else:
			self._step = 0
			self._loss = loss
		
		return False

def main():

	variable = sys.argv

	# 時刻tの地点数（入力データ数は１）
	num_input = 1

	# 中間層のユニット数（ハイパーパラメータのためチューニングする）
	num_hidden = int(variable[1])

	# RNNでさかのぼる時刻数
	LAG = 1

	# 時刻t+1の地点数（出力データ数は１）
	num_out = 1

	# havenデータを含める/含めない（1/0）
	haven_flag = 1

	train_population_data_size = 4 * 489	# トレーニングデータで扱う地点数（4日分）
	val_population_data_size = 1 * 489	# バリデーションデータで扱う地点数（１日分）

	epochs = 200000
	minibatch_size = 489	# 各エリアの地点数
	num_minibatch_train = int(train_population_data_size / float(minibatch_size))	# ミニバッチ数
	num_minibatch_val = int(val_population_data_size / float(minibatch_size))

	device = torch.device('cuda')
#	device = torch.device('cpu')

	model = Predictor(num_input, num_hidden, num_out, minibatch_size, device)
	model = model.to(device)

	criterion = torch.nn.MSELoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	early_stopping = EarlyStopping(patience=5000, verbose=1)

	train_input = torch.FloatTensor(np.load('train_input.npy')).to(device)
	val_input = torch.FloatTensor(np.load('val_input.npy')).to(device)
	train_target = torch.FloatTensor(np.load('train_target.npy')).to(device)
	val_target = torch.FloatTensor(np.load('val_target.npy')).to(device)

	train_squerrs = []
	val_squerrs = []
	rmse_val_best = float('inf') 

	params_train = []
	params_val = []
	
	for epoch in range(epochs):
		params_train.clear()
		params_val.clear()
		for i in range(num_minibatch_train):

			latlon_index = np.load('latlonindex489.npy')
	
			model.zero_grad()
			model.hidden = model.init_hidden(device)

			train_pred, param = model(LAG, train_input[:, i * minibatch_size:(i + 1) * minibatch_size, :], latlon_index, haven_flag, device)

			squerr = criterion(train_pred, train_target[:, i * minibatch_size:(i + 1) * minibatch_size, :])
			train_squerrs.append(squerr.item())

			optimizer.zero_grad()
			squerr.backward(retain_graph=True)

			optimizer.step()

			params_train.append(param)

		for j in range(num_minibatch_val):

			latlon_index = np.load('latlonindex489.npy')
	
			model.hidden = model.init_hidden(device)

			val_pred, param = model(LAG, val_input[:, j * minibatch_size:(j + 1) * minibatch_size, :], latlon_index, haven_flag, device)

			val_squerrs.append(criterion(val_pred, val_target[:, j * minibatch_size:(j + 1) * minibatch_size, :]).item())

			params_val.append(param)

		print('epoch = {}' .format(epoch))
		print('loss_val   = {}' .format(np.average(val_squerrs)))
		print('loss_train = {}' .format(np.average(train_squerrs)))

		train_squerrs.clear()

		if np.average(val_squerrs) < rmse_val_best: 
			torch.save(model, 'attention_%sunit' % (str(num_hidden)))
			rmse_val_best = np.average(val_squerrs)

		if early_stopping.validate(np.average(val_squerrs)):
			break

		val_squerrs.clear()

	np.save("params_train", params_train)
	np.save("params_val", params_val)

if __name__ == '__main__':
	main()
