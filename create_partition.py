import random as rn
import re

INPUT_FILE = open('./i2b2/combine.data','r')

per = 0.8

TRAIN_FILE_ENTTYPE = open('./i2b2/i2b2-80-enttype.train', 'w')
TEST_FILE_ENTTYPE = open('./i2b2/i2b2-20-enttype.test', 'w')
TRAIN_FILE = open('./i2b2/i2b2-80.train', 'w')
TEST_FILE = open('./i2b2/i2b2-20.test', 'w')

#Lists for samples containing sentences with entities replaced with entity types
other1 = []
tecp1 = []
trcp1 = []
trap1 = []
pip1 = []
terp1 = []
trnap1 = []
trwp1 = []
trip1 = []

#Lists for unaltered samples
other2 = []
tecp2 = []
trcp2 = []
trap2 = []
pip2 = []
terp2 = []
trnap2 = []
trwp2 = []
trip2 = []

samples = INPUT_FILE.read().strip().split('\n\n\n')

for i,sample in enumerate(samples):
	if(i%1000==0):
		print i,"samples read"
	name, sent, entities, relation = sample.strip().split('\n')	
	sent_length = len(sent.split())
	
	m = re.match(r"\(\[['\"](.*)['\"], (\d*), (\d*), '(.*)'\], \[['\"](.*)['\"], (\d*), (\d*), '(.*)'\]\)", entities.strip())
	if m :
		e1   = m.group(1)
		e1_s = int(m.group(2))
		e1_e = int(m.group(3))
		e1_t = m.group(4)

		e2   = m.group(5)
		e2_s = int(m.group(6))
		e2_e = int(m.group(7))
		e2_t = m.group(8)

	sent = re.sub(r"[-+]?\d*\.\d+|\d+","NUM",sent.lower())

	sent_enttype = sent.replace(e1,e1_t.upper()+'_A')
	sent_enttype = sent_enttype.replace(e2,e2_t.upper()+'_B')
	
	
	ma = re.match(r"\[['\"](.*)['\"], '(.*)', ['\"](.*)['\"]\]", relation.strip())
	if(ma):
		label = ma.group(2)		
	elif relation == '[0]':
		label = 'other'
	else:
		print ("Error in reading", relation)
		exit(0)

	if (label=="other"):
		other1.append([name,sent_enttype,sent_length,label])
		other2.append([name,sent,e1_t,e2_t,sent_length,label])
	elif (label=="TeCP"):
		tecp1.append([name,sent_enttype,sent_length,label])
		tecp2.append([name,sent,e1_t,e2_t,sent_length,label])
	elif (label=="TrCP"):
		trcp1.append([name,sent_enttype,sent_length,label])
		trcp2.append([name,sent,e1_t,e2_t,sent_length,label])
	elif (label=="TrAP"):
		trap1.append([name,sent_enttype,sent_length,label])
		trap2.append([name,sent,e1_t,e2_t,sent_length,label])
	elif (label=="PIP"):
		pip1.append([name,sent_enttype,sent_length,label])
		pip2.append([name,sent,e1_t,e2_t,sent_length,label])
	elif (label=="TeRP"):
		terp1.append([name,sent_enttype,sent_length,label])
		terp2.append([name,sent,e1_t,e2_t,sent_length,label])
	elif (label=="TrNAP"):
		trnap1.append([name,sent_enttype,sent_length,label])
		trnap2.append([name,sent,e1_t,e2_t,sent_length,label])
	elif (label=="TrWP"):
		trwp1.append([name,sent_enttype,sent_length,label])
		trwp2.append([name,sent,e1_t,e2_t,sent_length,label])
	elif (label=="TrIP"):
		trip1.append([name,sent_enttype,sent_length,label])
		trip2.append([name,sent,e1_t,e2_t,sent_length,label])


train1 = other1[:int(len(other1)*per)+1] + tecp1[:int(len(tecp1)*per)+1] + trcp1[:int(len(trcp1)*per)+1] + trap1[:int(len(trap1)*per)+1] + pip1[:int(len(pip1)*per)+1] + terp1[:int(len(terp1)*per)+1] + trnap1[:int(len(trnap1)*per)+1] + trwp1[:int(len(trwp1)*per)+1] + trip1[:int(len(trip1)*per)+1]
test1 = other1[int(len(other1)*per)+1:] + tecp1[int(len(tecp1)*per)+1:] + trcp1[int(len(trcp1)*per)+1:] + trap1[int(len(trap1)*per)+1:] + pip1[int(len(pip1)*per)+1:] + terp1[int(len(terp1)*per)+1:] + trnap1[int(len(trnap1)*per)+1:] + trwp1[int(len(trwp1)*per)+1:] + trip1[int(len(trip1)*per)+1:]

train2 = other2[:int(len(other2)*per)+1] + tecp2[:int(len(tecp2)*per)+1] + trcp2[:int(len(trcp2)*per)+1] + trap2[:int(len(trap2)*per)+1] + pip2[:int(len(pip2)*per)+1] + terp2[:int(len(terp2)*per)+1] + trnap2[:int(len(trnap2)*per)+1] + trwp2[:int(len(trwp2)*per)+1] + trip2[:int(len(trip2)*per)+1]
test2 = other2[int(len(other2)*per)+1:] + tecp2[int(len(tecp2)*per)+1:] + trcp2[int(len(trcp2)*per)+1:] + trap2[int(len(trap2)*per)+1:] + pip2[int(len(pip2)*per)+1:] + terp2[int(len(terp2)*per)+1:] + trnap2[int(len(trnap2)*per)+1:] + trwp2[int(len(trwp2)*per)+1:] + trip2[int(len(trip2)*per)+1:]

rn.shuffle(train1)
rn.shuffle(test1)
rn.shuffle(train2)
rn.shuffle(test2)

for t in train1:
	TRAIN_FILE_ENTTYPE.write(t[0]+'\n'+t[1]+'\n'+str(t[2])+'\n'+t[3]+'\n\n')

for t in test1:
	TEST_FILE_ENTTYPE.write(t[0]+'\n'+t[1]+'\n'+str(t[2])+'\n'+t[3]+'\n\n')

for t in train2:
	TRAIN_FILE.write(t[0]+'\n'+t[1]+'\n'+t[2]+','+t[3]+'\n'+str(t[4])+'\n'+t[5]+'\n\n')

for t in test2:
	TEST_FILE.write(t[0]+'\n'+t[1]+'\n'+t[2]+','+t[3]+'\n'+str(t[4])+'\n'+t[5]+'\n\n')