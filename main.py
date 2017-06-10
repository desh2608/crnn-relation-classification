import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
import pickle
import source.crnn_max as ModelSource

fp_correct = open('./results/crnn-max-all-features_correct.txt','wb')
fp_wrong = open('./results/crnn-max-all-features_wrong.txt','wb')
# fp = open('./results/attention_weights.txt','wb')


with open('./i2b2/i2b2-train2.pickle', 'rb') as handle:
	W = pickle.load(handle)
	Y_onehot = pickle.load(handle)
	wv = pickle.load(handle)
	word_list = pickle.load(handle)
	rev_word_list = pickle.load(handle)
	label_dict = pickle.load(handle)
	rev_label_dict = pickle.load(handle)

# with open('./ddi/ddi-train.pickle', 'rb') as handle:
# 	W = pickle.load(handle)
# 	Y_onehot = pickle.load(handle)
# 	wv = pickle.load(handle)
# 	word_list = pickle.load(handle)
# 	rev_word_list = pickle.load(handle)
# 	label_dict = pickle.load(handle)
# 	rev_label_dict = pickle.load(handle)

per = 0.8

num_total = len(W)
seq_len = len(W[0])
word_dict_size = len(word_list)
label_dict_size = len(label_dict)

# Splitting training set into train and validation set
# W_train = W[:int(per*num_total)+1]
# W_dev = W[int(per*num_total)+1:]
# Y_train = Y_onehot[:int(per*num_total)+1]
# Y_dev = Y_onehot[int(per*num_total)+1:]

W_train = W
Y_train = Y_onehot

model = ModelSource.Model(label_dict_size,seq_len,word_dict_size,wv)

## Training the model
num_train = len(W_train)
y_true_list = []
y_pred_list = []
num_epochs = 30
N = 5
batch_size = 256
num_batches_per_epoch = int(num_train/batch_size)


def test_step(W_te, Y_te):
	n = len(W_te)
	num = int(n/batch_size) + 1
	sample = []
	for batch_num in range(num):	
		start_index = batch_num*batch_size
		end_index = min((batch_num + 1) * batch_size, n)
		sample.append(range(start_index, end_index))
	pred = []
	for i in sample:
		p = model.test_step(W_te[i], Y_te[i])
		pred.extend(p)
	return pred


for j in range(num_epochs):
	acc = []		
	step = 0
	sam=[]
	for batch_num in range(num_batches_per_epoch):	
		start_index = batch_num*batch_size
		end_index = (batch_num + 1) * batch_size
		sam.append(range(start_index, end_index))		
	
	for rang in sam:
		step,acc_cur  = model.train_step(W_train[rang], Y_train[rang]) 
		acc.append(acc_cur)
	
	acc = np.array(acc)
	print "Average accuracy for epoch",j+1,"=",np.mean(acc)
	# if ((j+1)%N==0):
	# 	# fp.write('Epoch: '+str(j+1)+'\n')
	# 	pred = test_step(W_dev, d1_list_te, d2_list_te, Y_dev)
	# 	print "test data size ", len(pred)
	# 	y_true = np.argmax(Y_dev, 1)
	# 	y_pred = pred
	# 	# print(classification_report(y_true, y_pred,[1,2,3,4],digits=4))
	# 	# fp.write(classification_report(y_true, y_pred,[1,2,3,4,5,6,7,8],digits=4))
	# 	print(classification_report(y_true, y_pred,[1,2,3,4,5,6,7,8],digits=4))
print "Training finished."
# fp.close()
##------------------------------------------------------------------------------------##

##TESTING

with open('./ddi/ddi-test.pickle', 'rb') as handle:
	# sent_names = pickle.load(handle)
	sentences = pickle.load(handle)
	sent_lengths = pickle.load(handle)
	W_te = pickle.load(handle)
	Y_onehot = pickle.load(handle)
	wv = pickle.load(handle)
	word_list = pickle.load(handle)
	rev_word_list = pickle.load(handle)
	label_dict = pickle.load(handle)
	rev_label_dict = pickle.load(handle)

print "Test data loaded"

num_total = len(W_te)
seq_len = len(W_te[0])
# word_dict_size = len(word_list)
# label_dict_size = len(label_dict)

# att_weights = []
pred = test_step(W_te,Y_onehot)

y_true = np.argmax(Y_onehot, 1)
y_pred = pred
# print(classification_report(y_true, y_pred,[1,2,3,4],digits=4))
print(classification_report(y_true, y_pred,[1,2,3,4,5,6,7,8],digits=4))
# fp.write(classification_report(y_true, y_pred,[1,2,3,4,5,6,7,8],digits=4))
# i=0
# j=0
# atts = []
# for t,p in zip(y_true.tolist(),y_pred):
# 	sent = sentences[i]
# 	true_class = rev_label_dict[t]
# 	length = int(sent_lengths[i])
	# att = att_weights[i].tolist()
	# i+=1
	# if(t==p and t!=0):
	# 	# fp.write(str(j) + '\n'+ sent+'\n'+str(length) + '\n' +true_class+'\n\n')
	# 	j+=1
		# atts.append(att[:length])
# i = 0
# n = len(y_true)
# sum_correct = 0
# nc = 0
# sum_wrong = 0
# nw = 0
# for t,p in zip(y_true.tolist(),y_pred):
# 	name = sent_names[i]
# 	sent = sentences[i]
# 	length = sent_lengths[i]
# 	true_class = rev_label_dict[t]
# 	pred_class = rev_label_dict[p]
# 	i += 1
# 	if(t!=0):
# 		if(t==p):
# 			fp_correct.write(name+"\n"+sent+"\n"+str(length)+"\n"+true_class+"\n"+pred_class+"\n\n")
# 			sum_correct += int(length)
# 			nc += 1
# 		else:
# 			fp_wrong.write(name+"\n"+sent+"\n"+str(length)+"\n"+true_class+"\n"+pred_class+"\n\n")
# 			sum_wrong += int(length)
# 			nw += 1
	
# avg_correct = float(sum_correct)/nc
# avg_wrong = float(sum_wrong)/nw

# print "Average correct sentence length",avg_correct
# print "Average wrong sentence length",avg_wrong

# fp_correct.close()
# fp_wrong.close()
# fp.close()
