import sys, os
import matplotlib.pyplot as plt
colors = ['green', 'red', 'blue', 'yellow']


#####################################################################################
#Ensure log files have required data before running this code
#Use this as python log_to_plot.py <unique question_number(used to save the plot)>
#Before running initialise log_list dictionary below
#Make sure there is a directory named 'plots'

#######################################################################################
#Dictionary of {log folder name : Line name for legend in plot, ........}
#Each log folder should have a log_train.txt and a log_val.txt
#Initialize this before running. For example : 
log_list = {'log1' : Gradient Descent', 'log2' :'Momentum based Descent', 'log3' : 'NAG', 'log4' : 'Adam' }
# log_list = {'logq11' : '50 neurons', 'logq12' : '100 neurons', 'logq13' : '200 neurons', 'logq14' : '300 neurons'}

# log_list = {'logq41' : '50 neurons', logq42' : '100 neurons', 'logq43' : '200 neurons', 'logq44' : '300 neurons'}


# log_list = {'logq71' : 'Sigmoid', 'logq12' : 'Tanh'}


######################################################################################
os.system('mkdir plots')


fig = plt.figure()
i = 0
for (folder_name, line_name) in log_list.items():

	trainlog = open(folder_name +"/log_train.txt",'r').readlines()
	epochs_train = []
	losses_train = []
	for line in trainlog:
		infos = line.strip().split(',')
		epochs_train.append(int(infos[0].strip().split(' ')[1]))
		losses_train.append(float(infos[2].strip().split(' ')[1])/55000)
	
	plt.plot(epochs_train, losses_train, color=colors[i], linewidth=3, label = line_name)
	i += 1

plt.title('Training Loss vs Epochs')
plt.ylabel('Average Loss per data point')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('plots/' + sys.argv[1] +'_train.png')
# plt.show()








fig = plt.figure()
i = 0
for (folder_name, line_name) in log_list.items():

	vallog = open(folder_name + "/log_val.txt",'r').readlines()
	epochs_val = []
	losses_val = []
	for line in vallog:
		infos = line.strip().split(',')
		epochs_val.append(int(infos[0].strip().split(' ')[1]))
		losses_val.append(float(infos[2].strip().split(' ')[1])/5000)

	plt.plot(epochs_val, losses_val, color=colors[i], linewidth=3, label = line_name)
	i += 1

plt.title('Validation Loss vs Epochs')
plt.ylabel('Average Loss per data point')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('plots/' + sys.argv[1] +'_val.png')
# plt.show()


	
