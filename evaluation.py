
import os
import matplotlib.pyplot as plt
import numpy as np
import heapq


from nltk.translate.bleu_score import sentence_bleu

#Makes a prediction and returns 
def GreedyPrediction(model, modelAttention, x, h, length,oov_token,AttentionIndicator=True):

	cat_dim = len(h)

	X = np.zeros((x.shape[0],length))
	X[:,0] = x 
	
	AttentionValues = np.zeros((x.shape[0],length,cat_dim))
		
	for i in range(1,length):
			
		a = model.predict([h[0],h[1],h[2],h[3],X[:,0:i]])
		
		if AttentionIndicator:
			att = modelAttention.predict([h[0],h[1],h[2],h[3],X[:,0:i]])
			AttentionValues[:,i,:] = att
		
			
		for j in range(0,len(x)):
			max2 = heapq.nlargest(2, range(len(a[j])), a[j].take)
			
			if max2[0] == oov_token:
				X[j,i] = max2[1]	
			else:
				X[j,i] = max2[0]
	
	if AttentionIndicator:		
		return X,AttentionValues
	else:
		return X
				
def perplexityOld(model,Data,length):

	h = []
	
	for i in range(0,len(Data)-1):
		h.append(Data[i])
	
	x = Data[-1].flatten()
	

	X = np.zeros((x.shape[0],length))
	Y = np.ones(x.shape[0])
	X[:,0] = x 
	
	for i in range(1,length):
		a = model.predict([h[0],h[1],h[2],h[3],X[:,0:i]])
		
		X[:,i] = np.argmax(a,axis=1)		
		Y *= np.amax(a[:],1)
			
	Y = 1/Y
	Y = np.power(Y,1/length)
	
	return np.mean(Y)	
	
def perplexity(model,eos_token,input,y_true):



	prediction = model.predict(input)
	
	
	#print(y_true.shape)
	#print(input[0].shape)
	#print(input[-1].shape)
	
	Y = np.ones(prediction.shape[0])
	for i in range(0,len(input)):
		
		EndOfSentence = False
		helfer = 0
		for j in range(0,40):
			
			if EndOfSentence == False:
				helfer+=1
				Y[i] *= prediction[i,j,int(y_true[i,j])]
				if int(y_true[i,j]) == eos_token:
					EndOfSentence = True
		
		
		Y[i] = 1/Y[i]
		Y[i] = np.power(Y[i],1.0/helfer)
					
	return np.mean(Y)	

		
def PlotLosses(baseDirectory,KindOfLoss,iteration,
			TrainLoss,
			ValidationLoss,
			):
				
	if not os.path.exists(baseDirectory):
		os.makedirs(baseDirectory)

	directory = baseDirectory + 'epoch_' + str(iteration) + '.png'		
			
	plt.plot(np.arange(1,len(TrainLoss)+1),TrainLoss)
	plt.plot(np.arange(1,len(TrainLoss)+1),ValidationLoss)
	#plt.title('Model loss')
	plt.ylabel(KindOfLoss)
	plt.xlabel('Epoch')
	plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')	
	plt.savefig(directory)		
	plt.close()	
	
def PlotLossesWithLine(baseDirectory,KindOfLoss,iteration,
			TrainLoss,
			ValidationLoss,
			Limit,
			):
				
	if not os.path.exists(baseDirectory):
		os.makedirs(baseDirectory)

	directory = baseDirectory + 'epoch_' + str(iteration) + '.png'		
			
	plt.plot(TrainLoss)
	plt.plot(ValidationLoss)
	plt.plot(np.ones(len(ValidationLoss))*Limit,'r--')
	plt.title('Model loss')
	plt.ylabel(KindOfLoss)
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper right')	
	plt.savefig(directory)		
	plt.close()		
	
	
def pltAttention(value,x_ticks,y_ticks=('country','province','variety','price'),name='herbert'):	

	value = np.transpose(value)
	value = np.around(value, decimals=2)

	fig, ax = plt.subplots(2)

	ax[0].set_xticks(np.arange(20))#len(x_ticks.split()[0:20])))
	ax[0].set_yticks(np.arange(4))#np.arange(len(y_ticks)))
	# ... and label them with the respective list entries
	ax[0].set_xticklabels(x_ticks.split()[0:20])
	ax[0].set_yticklabels(y_ticks)

	plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")

	plt.setp(ax[0].get_yticklabels(), ha="right",
		rotation_mode="anchor")
	 
	for i in range(len(y_ticks)):
		for j in range(len(x_ticks.split()[0:20])):
			text = ax[0].text(j, i, value[i, j],
				ha="center", va="center", color="r",fontsize=8)
				
	ax[0].set_title("Attention")
	im = ax[0].imshow(value[:,0:20],cmap='gray',aspect='auto',
		vmin=0,
		vmax=1)
	#fig.tight_layout()
	#plt.show()	 
	
	
	ax[1].set_xticks(np.arange(20))#len(x_ticks.split()[20:])))
	ax[1].set_yticks(np.arange(4))#len(y_ticks)))
	# ... and label them with the respective list entries
	ax[1].set_xticklabels(x_ticks.split()[20:])
	ax[1].set_yticklabels(y_ticks)

	plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")

	plt.setp(ax[1].get_yticklabels(), ha="right",
		rotation_mode="anchor")
	 	 
	for i in range(len(y_ticks)):
		for j in range(len(x_ticks.split()[20:])):
			text = ax[1].text(j, i, value[i, j+20],
				ha="center", va="center", color="r",fontsize=8)
				
	#ax[1].set_title("Attention")
	im = ax[1].imshow(value[:,20:],cmap='gray',aspect='auto',
		vmin=0,
		vmax=1)
	
	fig.tight_layout()
	plt.savefig(name)
	plt.close()	
	
def ComputeSamples(baseDirectory,iteration, tokenizer,
			model, modelAttention, Data, length,oov_token,NumberOfSamplesComputed,AttentionIndicator=True):
	
	h = []
	
	for i in range(0,len(Data)-1):
		h.append(Data[i][0:NumberOfSamplesComputed])
	
	x = Data[-1][0:NumberOfSamplesComputed].flatten()
	
	
	
	if AttentionIndicator:
		X, Att = GreedyPrediction(model, modelAttention, x, h, length,oov_token,AttentionIndicator=AttentionIndicator)
	else:
		X = GreedyPrediction(model, modelAttention, x, h, length,oov_token,AttentionIndicator=AttentionIndicator)
	Sentences = tokenizer.sequences_to_texts(X)
	
	if not os.path.exists(baseDirectory+'samples//'):
		os.makedirs(baseDirectory+'samples//')
	
	with open(baseDirectory + 'samples//' + 'epoch_' + str(iteration) + '.txt', 'w') as f:
		for item in Sentences:
			f.write("%s\n" % item)
			
			
def ReturnComputedSamples(tokenizer, model, modelAttention, Data, length,oov_token):
	
	h = []
	
	for i in range(0,len(Data)-1):
		h.append(Data[i])
	
	x = Data[-1].flatten()
	
	X = GreedyPrediction(model, modelAttention, x, h, length,oov_token,AttentionIndicator=False)
	#Sentences = tokenizer.sequences_to_texts(X)
	
	return X

def WriteAsTextToFile(baseDirectory, X, tokenizer):
	Sentences = tokenizer.sequences_to_texts(X)
		
	with open(baseDirectory, 'w') as f:
		for item in Sentences:
			f.write("%s\n" % item)


		
def plotAttention(baseDirectory,iteration, tokenizer,model, modelAttention, Data, NumberOfSamplesComputed, length,oov_token):

	h = []
	
	for i in range(0,len(Data)-1):
		h.append(Data[i][0:NumberOfSamplesComputed])
	
	x = Data[-1][0:NumberOfSamplesComputed].flatten()


	X, Att = GreedyPrediction(model,modelAttention,
				x,h,length,oov_token,AttentionIndicator=True)
				
	text = tokenizer.sequences_to_texts(X)
	
	for reference in range(0,20):#len(X)):
		dir = baseDirectory + 'referenceNR' + str(reference) + '//'
		if not os.path.exists(dir):
			os.makedirs(dir)
		dir = dir + 'epoch_' + str(iteration) + '.png'
		pltAttention(Att[reference],text[reference],name=dir)
			
			
def plotAttention2(baseDirectory,iteration, tokenizer,model, modelAttention, Data, NumberOfSamplesComputed, length,oov_token):
	
	Att = modelAttention.predict(Data)
	Snt = model.predict(Data)
	
	
	for reference in range(0,10):
		dir = baseDirectory + 'referenceNR' + str(reference) + '//'
		if not os.path.exists(dir):
			os.makedirs(dir)
		dir = dir + 'sample.txt'	
		
		text = tokenizer.sequences_to_texts(Data[-1][reference:reference+1])
		
		with open(dir, 'w') as f:
			f.write(text[0])
		
		
		
		
	Snt = np.argmax(Snt,axis=-1)
	
	#text = tokenizer.sequences_to_texts(np.argmax(Snt,axis=-1))#,axis=-1)
	text = tokenizer.sequences_to_texts(Snt)
	
	for reference in range(0,10):
		dir = baseDirectory + 'referenceNR' + str(reference) + '//'
		if not os.path.exists(dir):
			os.makedirs(dir)
		dir = dir + 'epoch_' + str(iteration) + '.png'
		pltAttention(Att[reference],text[reference],name=dir)
	

def GreedyPrediction2(model, modelAttention, x, h, length,oov_token,KNN,AttentionIndicator=True):

	cat_dim = len(h)

	X = np.zeros((x.shape[0],length))
	X[:,0] = x 
	
	AttentionValues = np.zeros((x.shape[0],length,cat_dim))
		
	for i in range(1,length):
			
		a = model.predict([h[0],h[1],h[2],h[3],X[:,0:i]])
		
		a = KNN.predict(a)
		
		if AttentionIndicator:
			att = modelAttention.predict([h[0],h[1],h[2],h[3],X[:,0:i]])
			AttentionValues[:,i,:] = att
		
		
		#print(a)
		#print(a.shape)
		
		for j in range(0,len(x)):
			X[j,i] = a[j]
	
	if AttentionIndicator:		
		return X,AttentionValues
	else:
		return X
				

def ComputeSamples2(baseDirectory,iteration, tokenizer,
			model, modelAttention, Data, length,oov_token,NumberOfSamplesComputed,KNN,AttentionIndicator=True):
	
	h = []
	
	for i in range(0,len(Data)-1):
		h.append(Data[i][0:NumberOfSamplesComputed])
	
	x = Data[-1][0:NumberOfSamplesComputed].flatten()
	
	
	
	if AttentionIndicator:
		X, Att = GreedyPrediction2(model, modelAttention, x, h, length,oov_token,KNN,AttentionIndicator=AttentionIndicator)
	else:
		X = GreedyPrediction2(model, modelAttention, x, h, length,oov_token,KNN,AttentionIndicator=AttentionIndicator)
	Sentences = tokenizer.sequences_to_texts(X)
	
	if not os.path.exists(baseDirectory+'samples//'):
		os.makedirs(baseDirectory+'samples//')
	
	with open(baseDirectory + 'samples//' + 'epoch_' + str(iteration) + '.txt', 'w') as f:
		for item in Sentences:
			f.write("%s\n" % item)
			

def ComputeBleuScoresTraining(model,tokenizer,Data,References):

	NumberOfSentences = len(References)
	

	SentencesAsInt = np.argmax(model.predict(Data),axis=-1)

	Sentences = tokenizer.sequences_to_texts(SentencesAsInt)
	ReferenceSentences = tokenizer.sequences_to_texts(References)
	

	def getCurrent(Current):
		try: 
			EndTokenIndex = Current.index('qqe')
			CurrentSentence = Current[0:EndTokenIndex]
		except:
			CurrentSentence = Current
			
		return CurrentSentence.split()	
	
	bleu1 = 0
	bleu4 = 0
	for i in range(0,NumberOfSentences):
		
		
		CurrentSentence = getCurrent(Sentences[i])
		CurrentReference = getCurrent(ReferenceSentences[i])
		
		CurrentSentence[0] = "hallo"
		CurrentReference[0] = "hallo"
		
		
	
		bleu1 += sentence_bleu([CurrentReference], CurrentSentence, weights=(1, 0, 0, 0))
		bleu4 += sentence_bleu([CurrentReference], CurrentSentence)#, weights=(1, 1, 1, 1))
		
	bleu1 = bleu1 / NumberOfSentences
	bleu4 = bleu4 / NumberOfSentences
	
	return bleu1, bleu4


#modelAttention,
def getAttention(model,modelAttention,tokenizer,Data,References,vocab_size,max_seq_length,baseDirectory,iteration):

	NumberofHighest = 100

	NumberOfSentences = len(References)
	

	SentencesAsInt = np.argmax(model.predict(Data),axis=-1)
	AttentionMap = modelAttention.predict(Data)
	#AttentionMap = np.random.uniform(size=(NumberOfSentences,40,4))
	
	AttentionValues = np.zeros((vocab_size,5))

	for i in range(0,NumberOfSentences):
		for j in range(0,max_seq_length):
			WordAsInt = SentencesAsInt[i,j]
		
			AttentionValues[WordAsInt][0:4] = AttentionMap[i,j]
			AttentionValues[WordAsInt][4] += 1
			
	for i in range(0,vocab_size):
		if AttentionValues[i][4] > 0:
			AttentionValues[i] = AttentionValues[i] / AttentionValues[i][4]
	
	highest = []
	
	for i in range(0,4):
		highest.append(np.argpartition(AttentionValues[:,i], -NumberofHighest)[-NumberofHighest:])
	
	AsWords = tokenizer.sequences_to_texts(highest)

	if not os.path.exists(baseDirectory+'highestAttention//'):
		os.makedirs(baseDirectory+'highestAttention//')

	with open(baseDirectory + 'highestAttention//' + 'epoch_' + str(iteration) + '.txt', 'w') as f:
		for item in AsWords:
			f.write("%s\n" % item)		
			







	
	
	





