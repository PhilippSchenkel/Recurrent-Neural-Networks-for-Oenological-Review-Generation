

import numpy as np 
import pandas as pd 
from pandas import read_csv

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

from keras.utils import to_categorical


# load dataset

def getEmbeddingMatrix(tokenizer,embedding_dim,vocab_size,OOVToken):
	
	# load the whole embedding into memory
	embeddings_index = dict()
	f = open('data//glove.6B.' +str(embedding_dim) + 'd.txt',encoding="utf8")
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))
	
	# create a weight matrix for words in training docs
	embedding_matrix = np.zeros((vocab_size+1, embedding_dim))
	for word, i in tokenizer.word_index.items():
		
		#if word == OOVToken:
		
			##Out of word embedding as average of all embeddings
			#print(i)
		#	embedding_matrix[i] = np.array([-0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011,
		#								  0.33359843,  0.16045167,  0.03867431,  0.17833012,  0.04696583, -0.00285802,
		#								  0.29099807,  0.04613704, -0.20923874, -0.06613114, -0.06822549,  0.07665912,
		#								  0.3134014,   0.17848536, -0.1225775,  -0.09916984, -0.07495987,  0.06413227,
		##								  0.14441176,  0.60894334,  0.17463093,  0.05335403, -0.01273871,  0.03474107,
			#							 -0.8123879, -0.04688699,  0.20193407,  0.2031118,  -0.03935686,  0.06967544,
		#								 -0.01553638, -0.03405238, -0.06528071,  0.12250231,  0.13991883, -0.17446303,
		#								 -0.08011883,  0.0849521, -0.01041659, -0.13705009,  0.20127155,  0.10069408,
		#								  0.00653003,  0.01685157])
		
		if i <= vocab_size:
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
			else:
				print('Not in Gloves: ' + word)
				
				#embedding_matrix[i] = np.array([-0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011,
				#						  0.33359843,  0.16045167,  0.03867431,  0.17833012,  0.04696583, -0.00285802,
				#						  0.29099807,  0.04613704, -0.20923874, -0.06613114, -0.06822549,  0.07665912,
				#						  0.3134014,   0.17848536, -0.1225775,  -0.09916984, -0.07495987,  0.06413227,
				#						  0.14441176,  0.60894334,  0.17463093,  0.05335403, -0.01273871,  0.03474107,
				#						 -0.8123879, -0.04688699,  0.20193407,  0.2031118,  -0.03935686,  0.06967544,
				#						 -0.01553638, -0.03405238, -0.06528071,  0.12250231,  0.13991883, -0.17446303,
				#						 -0.08011883,  0.0849521, -0.01041659, -0.13705009,  0.20127155,  0.10069408,
				#						  0.00653003,  0.01685157])
	return embedding_matrix	

##Drop rows with invalid country values ('NAN')
def DropSomething(dataframe,column, what):

	toDrop = []
	for i in range(0,len(dataframe)):
		if not type(dataframe.at[i,column]) is what:
			toDrop.append(i)
	
	#dataframe = dataframe.drop(toDrop)
	
	return toDrop

##Returns 'country','province','variety','price' and desc as dataframes
def getSpecificColumns(nrows):

	if nrows == -1:
		data = read_csv("data//winemag-data_first150k.csv", sep=",", header=0)
	else:
		data = read_csv("data//winemag-data_first150k.csv", sep=",", header=0,nrows=nrows)
	##reduce amount of information
	data = data

	##Drop invalid rows
	toDrop = []
	toDrop+= DropSomething(data,'country',str)
	toDrop+=(DropSomething(data,'province',str))
	toDrop+=(DropSomething(data,'variety',str))
	data = data.dropna(subset=['price'])
	data = data.drop(toDrop)
	
	##Only get the reviews
	desc = data['description']
	desc = desc.str.lower()
	desc =desc.str.replace(".", " .")
	desc =desc.str.replace(",", "")
	
	desc =desc.str.replace("n't", " not")
	desc =desc.str.replace("'s,", " is")
	
	
	data['country'].str.replace(" ", "")
	data['province'].str.replace(" ", "")
	data['variety'].str.replace(" ", "")
	
	#print('unique values')
	#print(len(data['colour'].unique()))
	#print(len(data['country'].unique()))
	#print(len(data['province'].unique()))
	#print(len(data['variety'].unique()))


	return pd.DataFrame(data, columns=['country','province','variety','price']), desc

#Currently without usage
def AsInt(data):
	
	dataNew = data.copy()
	
	labels = data['country'].astype('category').cat.categories.tolist()
	replace_map_comp = {'country' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
	dataNew.replace(replace_map_comp, inplace=True)
	
	labels = data['province'].astype('category').cat.categories.tolist()
	replace_map_comp = {'province' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
	dataNew.replace(replace_map_comp, inplace=True)

	labels = data['variety'].astype('category').cat.categories.tolist()
	replace_map_comp = {'variety' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
	dataNew.replace(replace_map_comp, inplace=True)
	
	#labels = data['price'].astype('category').cat.categories.tolist()
	#replace_map_comp = {'price' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
	#dataNew.replace(replace_map_comp, inplace=True)

	print(dataNew.head())


def Tokenize(data,max_words):

	samples = data.tolist()
	
	newSamples = []

	for sample in samples:
		newSamples.append("qqs " + sample + " qqe")
		
	##To ensure that start and end token are in the tokenizer
	ensurance = ''
	for i in range(0,80):
		ensurance = ensurance + ' qqe ' + ' qqs '
	
	samples.append(ensurance)
	tokenizer = Tokenizer(num_words=max_words,oov_token='OOV')
	tokenizer.fit_on_texts(newSamples)

	##Text integer encoded as list
	sequences = tokenizer.texts_to_sequences(newSamples)

	##Text integer encoded as matrix
	x_train = np.zeros([len(sequences),len(max(sequences,key = lambda x: len(x)))])
	for i,j in enumerate(sequences):
		x_train[i][0:len(j)] = j
	
	#print(newSamples[0])
	#print(x_train[0])
	
	return x_train, tokenizer

	
##Returns a list of pandas dataframe containing one hot encoded categories, the sequences as integer encoding, the corresponding tokenizer
def getData(max_words,nrows=-1):
	data, desc = getSpecificColumns(nrows)
	
	x_train, tok = Tokenize(desc,max_words)

	##OneHotEncoding
	attributes = []
	
	for x in ['country', 'province', 'variety']:
		attributes.append(pd.get_dummies(data[x]).to_numpy())

	helfer = data['price'].to_numpy()
	
	from sklearn.preprocessing import KBinsDiscretizer

	qt = KBinsDiscretizer(n_bins=10,encode='onehot-dense')#ordinal')#, random_state=0)
	
	print(np.quantile(helfer.reshape(-1,1),[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
	
	
	helfer = qt.fit_transform(helfer.reshape(-1,1))

	
	attributes.append(helfer)

	data = pd.get_dummies(data)
	
	return attributes, x_train, tok

	
if __name__=="__main__":
	
	import matplotlib.pyplot as plt
	
	vocab_size = -1
	#attributes, sequences, tokenizer = Pipeline(vocab_size)
	
	attributes, sequences, tokenizer = getData(vocab_size,nrows=-1)
	
	
	wordcount = tokenizer.word_counts
	
	import operator
	sorted_words = sorted(tokenizer.word_counts.items(), key=operator.itemgetter(1),reverse=True)
	
	print('vocab size')
	print(len(sorted_words))
	
	
	
	frequencies = np.zeros(len(sorted_words))
	
	Threshhold = 1000
	Coverage = 0
	SumWords = 0
	
	Reached = False
	AtLeast100 = 0
	
	
	
	for i in range(0,len(sorted_words)):
	
		frequencies[i]= sorted_words[i][1]
	
		if frequencies[i] < Threshhold:
			if Reached == False:
				Reached = True
				print('Threshhold reached at: ' + str(i))
				AtLeast100 = i
	
	#SumWords = np.sum(frequencies)
	
	#Coverage = np.zeros(len(sorted_words))
	
	#percentage = 0.95
	#Reached = False
	#value = -1
	#for i in range(1,len(sorted_words)):
	#	Coverage[i] = np.sum(frequencies[0:i]) / SumWords
		
	#	if Coverage[i] > 0.99:
	#		if Reached == False:
	#			Reached = True
	#			value = i
	#	if i == AtLeast100:
	#		print('Words that appear at least ' + str(Threshhold) + ' times cover ' + str(Coverage[i]) + ' of the text')
	
		
	#print(value)
	
	for i in range(0,len(sorted_words)):
		if frequencies[i] < 10:
			print('Threshhold')
			print(i)
			break
	
	
	plt.bar(np.arange(0,len(sorted_words)),frequencies, edgecolor = "none")
	plt.plot(np.arange(0,len(sorted_words)),np.ones(len(sorted_words))*10,color='red', linestyle='dashed', label='Cut off Threshhold')
	#plt.plot((np.ones(np.max(frequencies))*9940).dtype('int'),np.arange(0,np.max(frequencies).dtype('int')),color='red', linestyle='dashed')
	plt.legend(loc="upper right")
	
	plt.title('Word frequencies')
	plt.yscale('log')
	plt.show()
	plt.close()
	plt.plot(Coverage)
	plt.show()
	plt.close()
	
	
	
	
	