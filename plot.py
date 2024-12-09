import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


legends = ['Non-Equivariant', 'Equivariant', '', '', '']

# add any folder directories here!
log_list = [
pd.read_csv("trained_models/no_equi_no_pred_5_orca/progress.csv"),
pd.read_csv("trained_models/equi_orca_5_no_history/progress.csv"),
	]


logDicts = {}
for i in range(len(log_list)):
	logDicts[i] = log_list[i]

graphDicts={0:'eprewmean', 1:'loss/value_loss'}

legendList=[]
# summarize history for accuracy

steps = 400

# for each metric
for i in range(len(graphDicts)):
	plt.figure(i)
	plt.title(graphDicts[i])
	j = 0

	timesteps = []
	values = []

	for key in logDicts:
		if graphDicts[i] not in logDicts[key]:
			continue
		else:
			plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
			ts = logDicts[key]['misc/total_timesteps'].values
			vs = logDicts[key][graphDicts[i]].values
			print(ts.shape[0])
			plt.plot(ts[:steps],vs[:steps])

			legendList.append(legends[j])
			print('avg', str(key), graphDicts[i], np.average(logDicts[key][graphDicts[i]]))
		j = j + 1
	print('------------------------')

	plt.xlabel('total_timesteps')
	plt.legend(legendList, loc='lower right')
	legendList=[]

plt.show()
