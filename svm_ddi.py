from sklearn.feature_extraction import DictVectorizer
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from nltk.tokenize import WordPunctTokenizer
from sklearn.metrics import classification_report
import nltk
import numpy as np
import re
import pickle
from geniatagger import GeniaTagger
tagger = GeniaTagger('/home/desh/geniatagger-3.0.2/geniatagger')

tokenizer = WordPunctTokenizer()


def preProcess(sent):
	sent = re.sub(r"(dg)+","num",sent.lower())
	sent = tokenizer.tokenize(sent)
	sent = ' '.join(sent)
	sent_list,_,_,_,_ = zip(*tagger.parse(sent)) 
	sent = ' '.join(sent_list)
	return sent

def find_sub_list(sl,l):
	sll=len(sl)
	for ind in (i for i,e in enumerate(l) if e==sl[0]):
		if l[ind:ind+sll]==sl:
			return ind,ind+sll-1

def dataRead(fname):
	print "Input File Reading"
	fp = open(fname, 'r')
	samples = fp.read().strip().split('\n\n')
	sent_lengths   = []		#1-d array
	sent_contents  = []		#2-d array [[w1,w2,....] ...]
	sent_lables    = []		#1-d array
	entity1_list   = []		#2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
	entity2_list   = []		#2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
	for sample in samples:
		sent, ent1, relation = sample.strip().split('\n')
	
		if len(sent.split()) > 100:
			continue

		sent_lables.append(relation)
		sent_contents.append(sent)
 
		e1 = sent.split().index('DRUGA')
		e2 = sent.split().index('DRUGB')
		
		entity1_list.append(e1)
		entity2_list.append(e2)
 
	return sent_contents, entity1_list, entity2_list, sent_lables 

def makeFeatures(sent_list, entity1_list, entity2_list):
	print 'Making Features'
	word_list = []
	d1_list = []
	d2_list = []
	type_list = []
	pos_list = []
	chunk_list = []
	for sent, ent1, ent2 in zip(sent_list, entity1_list, entity2_list):
		sent = preProcess(sent)
		sent_list1, _, pos_list1, chunk_list1, _ = zip(*tagger.parse(sent))

		# distance1 feature	
		d1 = []
		for i in range(len(sent_list1)):
			d1.append(abs(i-ent1))
		#distance2 feature		
		d2 = []
		for i in range(len(sent_list1)):
			d2.append(abs(i-ent2))

		word_list.append(sent_list1)
		pos_list.append(pos_list1)
		chunk_list.append(chunk_list1)
		d1_list.append(d1)
		d2_list.append(d2)
		# type_list.append(t) 
	
	return word_list, pos_list, chunk_list, d1_list, d2_list, type_list


def makeVector(sent_list, entity1_list, entity2_list, pos_list, chunk_list):
	features = []
	#['her amiodarone', 10, 11, 'treatment']
	for sent, ent1, ent2, pos, chunk in zip(sent_list, entity1_list, entity2_list, pos_list, chunk_list):
		sent = sent.split()
		sentf = {}
		#---------------------Context Features------------------------------------------

		#CF1 : any word between relation arguments
		for k,i in enumerate(range(ent1+1, ent2)):
			sentf["CF1_"+str(k)] = sent[i] 
		
		#CF2 : any pos between relation arguments
		for k,i in enumerate(range(ent1+1, ent2)):
			sentf["CF2_"+str(k)] = pos[i] 
		
		#CF3 : any bigram between relation arguments
		sent_bigrams = list(nltk.bigrams(sent[ent1+1:ent2]))
		for k,bigram in enumerate(sent_bigrams):
			sentf['CF3_'+str(k)] = '-'.join(bigram)
		
		#CF4 : word preciding first argument
		if ent1 == 0:
			sentf['CF4'] = '<S>'
		else:
			sentf['CF4'] = sent[ent1 - 1]

		#CF5 : word prediding second arguments
		if ent2 == 0:
			sentf['CF5'] = '<S>'
		else:
			sentf['CF5'] = sent[ent2 - 1]

		#CF6 : any three words succeeding the first arguments
		
		if ent1 <= len(sent) - 3 : 
			sent_trigrams = list(nltk.trigrams(sent[ent1+1:]))
			for k,trigram in enumerate(sent_trigrams):
				sentf['CF6_'+str(k)] = '-'.join(trigram)
		else:
			sentf['CF6_0'] = '<E>'
		
		#CF7 : any three succeeding the second arguments
		if ent2 <= len(sent) - 3 : 
			sent_trigrams = list(nltk.trigrams(sent[ent2+1:]))
			for k,trigram in enumerate(sent_trigrams):
				sentf['CF7_'+str(k)] = '-'.join(trigram)
		else:
			sentf['CF7_0'] = '<E>'

		#CF8 : sequence of chunk type between relation argumemts
		sentf['CF8'] = '-'.join(chunk[ent1+1:ent2])

		#CF9 : string of words between relation arguments
		sentf['CF9'] = '-'.join(sent[ent1+1:ent2])

		#CF13 : Distance between two arguments
		sentf['CF13'] = abs(sent.index('DRUGA') - sent.index('DRUGB'))

		#CF14 : Presence of puncuation sign between arguments
		if sent[ent1 : ent2] == [','] or ['and'] or ['or'] or ['/'] :
			sentf['CF14'] = True
		else:
			sentf['CF14'] = False

		features.append(sentf)
 
	return features		
	

ftrain = './ddi/train_data.txt'
tr_sent_contents, tr_entity1_list, tr_entity2_list, tr_sent_lables = dataRead(ftrain)

ftest = './ddi/test_data.txt'
te_sent_contents, te_entity1_list, te_entity2_list, te_sent_lables = dataRead(ftest)

# label_dict = {'other':0, 'TrWP': 1, 'TeCP': 2, 'TrCP': 3, 'TrNAP': 4, 'TrAP': 5, 'PIP': 6, 'TrIP': 7, 'TeRP': 8}
label_dict = {'false':0, 'mechanism': 1, 'effect': 2, 'advise': 3, 'int': 4}

#print X_train
Y_train = [label_dict[i] for i in tr_sent_lables]
Y_train = np.array(Y_train)

Y_test = [label_dict[i] for i in te_sent_lables]
Y_test = np.array(Y_test)

tr_sent_list, tr_pos_list, tr_chunk_list, tr_d1_list, tr_d2_list, tr_type_list = makeFeatures(tr_sent_contents, tr_entity1_list, tr_entity2_list)

tr_features = makeVector(tr_sent_contents, tr_entity1_list, tr_entity2_list, tr_pos_list, tr_chunk_list)

te_sent_list, te_pos_list, te_chunk_list, te_d1_list, te_d2_list, te_type_list = makeFeatures(te_sent_contents, te_entity1_list, te_entity2_list)

te_features = makeVector(te_sent_contents, te_entity1_list, te_entity2_list, te_pos_list, te_chunk_list)

print "len X_train before", len(tr_features)
print "len X_test before", len(te_features)

#features = tr_features + te_features

vec = DictVectorizer()

train = vec.fit_transform(tr_features)#.toarray()

features_name = vec.get_feature_names()

print "number of features", len(features_name)


# label_dict = {'other':0, 'TrWP': 1, 'TeCP': 2, 'TrCP': 3, 'TrNAP': 4, 'TrAP': 5, 'PIP': 6, 'TrIP': 7, 'TeRP': 8}
label_dict = {'false':0, 'mechanism': 1, 'effect': 2, 'advise': 3, 'int': 4}


X_train = train
X_test = vec.transform(te_features)

print "X_train after",X_train.shape
print "X_test after",X_test.shape


acc_list = []
clf = svm.SVC(kernel='linear', C=0.1).fit(X_train, Y_train)

a = clf.score(X_test, Y_test)

print "accuracy", a

y_true = Y_test
y_pred = clf.predict(X_test)

print(classification_report(y_true, y_pred,[1,2,3,4],digits=4))
