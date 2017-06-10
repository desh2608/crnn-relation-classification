import numpy as np
import pickle

def readData(fp):
	samples = fp.read().strip().split('\n\n')

	sent_names = []
	sentences = []
	sent_contents = []
	sent_labels = []
	sent_lengths = []

	for sample in samples:
		name,sent,length,label = sample.strip().split('\n')
		sent_names.append(name)
		sent_lengths.append(length)
		sent_labels.append(label)
		sentences.append(sent)
		sent_contents.append(sent.split())

	maxl = max([len(sent) for sent in sent_contents])
	pad_symbol= '<pad>'
	sent_padded = []

 	for sent in sent_contents:
		sent_new = []
		length = len(sent)
		for i in range(length):
			sent_new.append(sent[i])
		for i in range(length,maxl):
			sent_new.append(pad_symbol)
		sent_padded.append(sent_new)

	return sent_names,sent_padded,sent_lengths,sent_labels,sentences,maxl


def makeWordList(sent_list):
	wf = {}	#Word frequency
	for sent in sent_list:
		for w in sent:
 			if w in wf:
				wf[w] += 1
			else:
				wf[w] = 0
	wl = {}	#Word list
	rwl = {} #reverse Word List
	i = 0
	for w,f in wf.iteritems():		
		wl[w] = i
		rwl[i] = w
		i += 1
	wl['UNK'] = i
	return wl,rwl

def mapWordToId(sent_contents, word_dict):
	T = []
	for sent in sent_contents:
		t = []
		for w in sent:
			if w in word_dict:
				t.append(word_dict[w])
			else:
				t.append(word_dict['UNK'])
		T.append(t)
	return T

def readWordEmb(word_dict, fname, embSize=100):
	print "Reading word vectors"
	wv = []
	wl = []
	with open(fname, 'r') as f:
		for line in f :			
			vs = line.split()
			if len(vs) < 50 :
				continue
			vect = map(float, vs[1:])
			wv.append(vect)
			wl.append(vs[0])
	wordemb = []
	count = 0
	for word, id in word_dict.iteritems():
		if str(word) in wl:
			wordemb.append(wv[wl.index(str(word))])
		else:
			count += 1
			wordemb.append(np.random.rand(embSize))
	wordemb = np.asarray(wordemb, dtype='float32')
	print "Number of unknown word in word embedding", count
	return wordemb



emb_file = './i2b2/glove_100d_w9_pubmed.txt'
label_dict = {'other':0, 'TrWP': 1, 'TeCP': 2, 'TrCP': 3, 'TrNAP': 4, 'TrAP': 5, 'PIP': 6, 'TrIP': 7, 'TeRP': 8}
rev_label_dict = {0:'other', 1:'TrWP', 2:'TeCP', 3:'TrCP', 4:'TrNAP', 5:'TrAP', 6:'PIP', 7:'TrIP', 8:'TeRP'}

##Processing the training set
fp_train = open('./i2b2/80-20/i2b2-80-enttype.train','r')
sent_names,sent_padded,sent_lengths,sent_labels,sentences,seq_len = readData(fp_train)
fp_train.close()

word_list, rev_word_list = makeWordList(sent_padded)
sent_contents = mapWordToId(sent_padded, word_list)
wv = readWordEmb(word_list, emb_file)	

W =  np.array(sent_contents)
Y = [label_dict[label] for label in sent_labels]
Y_onehot = np.zeros((len(Y), len(label_dict)))

for i in range(len(Y)):
	Y_onehot[i][Y[i]] = 1

with open('./i2b2/i2b2-train.pickle', 'wb') as handle:
	pickle.dump(W, handle)
	pickle.dump(Y_onehot, handle)
	pickle.dump(wv, handle)
	pickle.dump(word_list, handle)
	pickle.dump(rev_word_list,handle)
	pickle.dump(label_dict, handle)
	pickle.dump(rev_label_dict,handle)


##Processing the test set
fp_test = open('./i2b2/80-20/i2b2-20-enttype.test','r')
sent_names,sent_padded,sent_lengths,sent_labels,sentences,seq_len = readData(fp_test)
fp_test.close()

word_list, rev_word_list = makeWordList(sent_padded)
sent_contents = mapWordToId(sent_padded, word_list)
wv = readWordEmb(word_list, emb_file)

W =  np.array(sent_contents)
Y = [label_dict[label] for label in sent_labels]
Y_onehot = np.zeros((len(Y), len(label_dict)))

for i in range(len(Y)):
	Y_onehot[i][Y[i]] = 1

with open('./i2b2/i2b2-test.pickle', 'wb') as handle:
	pickle.dump(sent_names,handle)
	pickle.dump(sentences,handle)
	pickle.dump(sent_lengths,handle)
	pickle.dump(W, handle)
	pickle.dump(Y_onehot, handle)
	pickle.dump(wv, handle)
	pickle.dump(word_list, handle)
	pickle.dump(rev_word_list,handle)
	pickle.dump(label_dict, handle)
	pickle.dump(rev_label_dict,handle)
