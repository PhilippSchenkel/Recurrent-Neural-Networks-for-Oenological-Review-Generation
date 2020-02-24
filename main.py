
import tensorflow as tf 

#tf.enable_eager_execution()

from preprocessing import getData
from preprocessing import getEmbeddingMatrix

#from losses import weighted_categorical_crossentropy

from evaluation import perplexity
from evaluation import PlotLosses
from evaluation import ComputeSamples
from evaluation import plotAttention
from evaluation import plotAttention2
from evaluation import WriteAsTextToFile
from evaluation import ReturnComputedSamples
from evaluation import PlotLossesWithLine
from evaluation import ComputeBleuScoresTraining
from evaluation import getAttention

#from classifier import DefineClassifier


from contextlib import redirect_stdout


from sklearn.model_selection import train_test_split

from keras.optimizers import RMSprop, Adam


import numpy as np
import os

#from model import getModel
#from modelDong import getModel
#from models.modelNoAttention import getModel
from models.modelDong import getModel
#from models.modelNoAttention import getModel

from keras.models import load_model
from keras.metrics import categorical_accuracy
	
from sklearn.metrics import mean_squared_error


import keras.backend as K 


#attentionType = 'NoAttention5'

#attentionType = 'NoAttention1'
attentionType = 'DongAgain3'
#attentionType = 'lalala'
AttentionIndicator = True
#AttentionIndicator = True

LoadModel = False



category_names = ['country','province','variety','price']
#Classifier = load_model('classifierModel/classifier.h5')
classifierLossValues = [0.79,0.64,0.52,0.000324]


vocab_size = 9940#35761
evaluation_frequency = 10

embedding_dim = 300
batch_size = 1500#00
mini_batch_size = 128#512
train_seq_length = 40

val_split = 0.01
test_split = 0.01

OOVToken = 'OOV'
NumberOfSamplesComputed = 20#50

#Data work
categoryData, sequenceData, tokenizer = getData(vocab_size,nrows=-1)



#print(categoryData[-1].shape)
#for i in range(0,4):
	#print(np.sum(categoryData[i],axis=0))
	#print('class ' + str(i))
	#numClasses = np.sum(categoryData[i],axis=0)
	#print(numClasses)
	#print('mean')
	#print(np.mean(numClasses))
	#print('median')
	#print(np.median(numClasses))
	#print('min')
	#print(np.min(numClasses))
	#print('max')
	#print(np.max(numClasses))

oov_token = tokenizer.word_index[OOVToken]

eos_token = tokenizer.word_index['qqe']


NumberOfSamples = len(sequenceData)
NumberOfCategories = len(categoryData)

categories_shapes = np.zeros(NumberOfCategories).astype('int')
for i in range(0,NumberOfCategories):
	categories_shapes[i] = categoryData[i].shape[1]

	
TrainingIndices = np.arange(NumberOfSamples)

TrainingIndices, ValidationIndices = train_test_split(TrainingIndices, test_size=val_split, random_state=42)
TrainingIndices, TestIndices = train_test_split(TrainingIndices, test_size=test_split, random_state=42)


dada = tokenizer.sequences_to_texts(sequenceData[TrainingIndices[0:20]])

with open('conf.txt', 'w') as f:
	for item in dada:
		f.write("%s\n" % item)	
		
dada = tokenizer.sequences_to_texts(sequenceData[ValidationIndices[0:20]])

with open('conf2.txt', 'w') as f:
	for item in dada:
		f.write("%s\n" % item)		

print(TrainingIndices[0:20])
		
print(a)

	
EmbeddingMatrix = getEmbeddingMatrix(tokenizer,embedding_dim,vocab_size,OOVToken)	
modelTrainingText, modelTrainingAttention, modelPredictionText, modelPredictionAttention = getModel(categories_shapes,EmbeddingMatrix,train_seq_length,True)


opti = RMSprop(0.002,clipvalue=5)
loss_weights = np.ones(vocab_size+1)
loss_weights[oov_token] = 0.01

modelTrainingText.compile(opti, loss='sparse_categorical_crossentropy',metrics=['acc'])# weighted_categorical_crossentropy(loss_weights))
modelTrainingText.summary()

def getDataBatch(indicesToPickFrom,categoryData,sequenceData,train_seq_length,classifier=False):
	
	X_batch = []
	for category in categoryData:
		X_batch.append(category[indicesToPickFrom])
	X_batch.append(sequenceData[indicesToPickFrom,0:train_seq_length])
	
	#Y_batch = getOneHotEncoding(indicesToPickFrom)
	#Y_batch = Y_batch[:,1:train_seq_length+1]	
		
	if classifier == False:		
		return X_batch#,Y_batch
	else:
		return X_batch[-1],X_batch[0:-1]
				
	
def getOneHotEncoding(Indices):

	sequences_one_hot = np.zeros((len(Indices),sequenceData.shape[1],vocab_size+1))
	
	for i in range(0,len(Indices)):
		for j in range(0,sequenceData.shape[1]):
			sequences_one_hot[i][j][int(sequenceData[Indices[i]][j])] = 1

	return sequences_one_hot

baseDirectory = 'results2//results' + attentionType + str(vocab_size) + '//'
if not os.path.exists(baseDirectory):
	os.makedirs(baseDirectory)	
	
with open(baseDirectory + 'modelPredictionTextSummary.txt', 'w') as f:
    with redirect_stdout(f):
        modelPredictionText.summary()
		
if LoadModel:
	modelTrainingText.load_weights(baseDirectory+'modelPredictionText_weights.h5')		

train_loss = []
validation_loss = []	

train_perplexity = []
validation_perplexity = []	


train_bleu1 = []
validation_bleu1 = []	

train_bleu4 = []
validation_bleu4 = []	

	
#X_val,y_val = getDataBatch(ValidationIndices,categoryData,sequenceData,train_seq_length)	

X_train_evaluation = getDataBatch(TrainingIndices[0:20],categoryData,sequenceData,1)	#, y_train_evaluation	
X_val_evaluation = getDataBatch(ValidationIndices[0:20],categoryData,sequenceData,1)	#	, y_val_evaluation
X_test_evaluation = getDataBatch(TestIndices,categoryData,sequenceData,1)	#, y_test_evaluation

X_train_evaluation2= getDataBatch(TrainingIndices[0:10],categoryData,sequenceData,train_seq_length)	#	, y_train_evaluation2 
X_val_evaluation2 = getDataBatch(ValidationIndices[0:10],categoryData,sequenceData,train_seq_length)		#, y_val_evaluation2
X_test_evaluation2 = getDataBatch(TestIndices[0:10],categoryData,sequenceData,train_seq_length)	#, y_test_evaluation2

classifier_train_losses = []
classifier_validation_losses = []


#for i in range(0,10):
#	X_train_evaluation.same.append(np.repeatcategoryData[0,TrainingIndices[0])
#	X_train_evaluation.same.append(categoryData[1,TrainingIndices[0])
#	X_train_evaluation.same.append(categoryData[2,TrainingIndices[0])
#	helfer = np.zeros(10)
#	helfer[i] = 1
#	X_train_evaluation.same.append(helfer)
		
#X_train_evaluation_same = getDataBatch(TrainingIndices[0:1],categoryData,sequenceData,1)


for i in range(0,NumberOfCategories):
	classifier_train_losses.append([])
	classifier_validation_losses.append([])



#TrainingIndices = TrainingIndices[0:1000]
for batch_iteration in range(0,200):
	
	history = modelTrainingText.fit(
		[categoryData[0][TrainingIndices],categoryData[1][TrainingIndices],categoryData[2][TrainingIndices],categoryData[3][TrainingIndices],sequenceData[TrainingIndices,0:train_seq_length]],
		sequenceData[TrainingIndices,1:train_seq_length+1].reshape((-1,train_seq_length,1)),
		validation_data = (
		[categoryData[0][ValidationIndices],categoryData[1][ValidationIndices],categoryData[2][ValidationIndices],categoryData[3][ValidationIndices],sequenceData[ValidationIndices,0:train_seq_length]],
		sequenceData[ValidationIndices,1:train_seq_length+1].reshape((-1,train_seq_length,1))),
		epochs=1,
		batch_size=mini_batch_size)
	
	train_loss += history.history['loss']
	validation_loss += history.history['val_loss']
	
	train_perplexity.append(perplexity(
			modelTrainingText,
			eos_token,
			[categoryData[0][TrainingIndices[0:1000]],categoryData[1][TrainingIndices[0:1000]],categoryData[2][TrainingIndices[0:1000]],categoryData[3][TrainingIndices[0:1000]],sequenceData[TrainingIndices[0:1000],0:train_seq_length]],
			sequenceData[TrainingIndices[0:1000],1:train_seq_length+1].reshape((-1,train_seq_length,1))))
			
			
	validation_perplexity.append(
			perplexity(
			modelTrainingText,
			eos_token,
			[categoryData[0][ValidationIndices],categoryData[1][ValidationIndices],categoryData[2][ValidationIndices],categoryData[3][ValidationIndices],sequenceData[ValidationIndices,0:train_seq_length]],
			sequenceData[ValidationIndices,1:train_seq_length+1].reshape((-1,train_seq_length,1))))
	
	
	bleu1, bleu4 = ComputeBleuScoresTraining(modelTrainingText,tokenizer,
		[categoryData[0][TrainingIndices[0:1000]],categoryData[1][TrainingIndices[0:1000]],categoryData[2][TrainingIndices[0:1000]],categoryData[3][TrainingIndices[0:1000]],sequenceData[TrainingIndices[0:1000],0:train_seq_length]],
		sequenceData[TrainingIndices[0:1000],1:train_seq_length+1])

	train_bleu1.append(bleu1)
	train_bleu4.append(bleu4)

	bleu1, bleu4 = ComputeBleuScoresTraining(modelTrainingText,tokenizer,
		[categoryData[0][ValidationIndices],categoryData[1][ValidationIndices],categoryData[2][ValidationIndices],categoryData[3][ValidationIndices],sequenceData[ValidationIndices,0:train_seq_length]],
		sequenceData[ValidationIndices,1:train_seq_length+1])
	
	validation_bleu1.append(bleu1)
	validation_bleu4.append(bleu4)


	DumpDirectory = 'results2//results' + attentionType + str(vocab_size) + '//dump//'
	if not os.path.exists(DumpDirectory):
		os.makedirs(DumpDirectory)	
		
	with open(DumpDirectory+'train_perplexity.txt', 'w') as f:
		for item in train_perplexity:
			f.write("%s\n" % item)	
			
	with open(DumpDirectory+'validation_perplexity.txt', 'w') as f:
		for item in validation_perplexity:
			f.write("%s\n" % item)	
	
	with open(DumpDirectory+'train_bleu1.txt', 'w') as f:
		for item in train_bleu1:
			f.write("%s\n" % item)	
	
	with open(DumpDirectory+'validation_bleu1.txt', 'w') as f:
		for item in validation_bleu1:
			f.write("%s\n" % item)	
			
	with open(DumpDirectory+'train_bleu4.txt', 'w') as f:
		for item in train_bleu4:
			f.write("%s\n" % item)	
	
	with open(DumpDirectory+'validation_bleu4.txt', 'w') as f:
		for item in validation_bleu4:
			f.write("%s\n" % item)	
			
	with open(DumpDirectory+'train_loss.txt', 'w') as f:
		for item in train_loss:
			f.write("%s\n" % item)	
			
	with open(DumpDirectory+'//validation_loss.txt', 'w') as f:
		for item in validation_loss:
			f.write("%s\n" % item)	




	
	if batch_iteration % evaluation_frequency == 0:
	
		#modelPredictionText.save_weights(baseDirectory+'modelPredictionText_weights.h5')
		
		
		#Evaluation
		modelTrainingAttention.set_weights(modelTrainingText.get_weights())
		
		modelPredictionText.set_weights(modelTrainingText.get_weights())
		modelPredictionAttention.set_weights(modelTrainingText.get_weights())
		
		
		
		##Error
		PlotLosses(baseDirectory + 'loss//','Average Categorical Crossentropy',batch_iteration,train_loss,validation_loss)
		
		##Perplexity	
		PlotLosses(baseDirectory+'perplexity//','perplexity',batch_iteration,train_perplexity,validation_perplexity)
		
		
		PlotLosses(baseDirectory+'bleu1//','BLEU 1',batch_iteration,train_bleu1,validation_bleu1)
		
		PlotLosses(baseDirectory+'bleu4//','BLEU 4',batch_iteration,train_bleu4,validation_bleu4)
		
			
		##Samples
		ComputeSamples(baseDirectory+'train_',batch_iteration, tokenizer,
				modelPredictionText, modelPredictionAttention, 
				X_train_evaluation,
				train_seq_length,
				oov_token,
				NumberOfSamplesComputed,
				AttentionIndicator
				)
		ComputeSamples(baseDirectory+'validation_',batch_iteration, tokenizer,
				modelPredictionText, modelPredictionAttention, 
				X_val_evaluation,
				train_seq_length,
				oov_token,
				NumberOfSamplesComputed,
				AttentionIndicator
				)			
		#ComputeSamples(baseDirectory+'test_',batch_iteration, tokenizer,
		#		modelPredictionText, modelPredictionAttention, 
		#		X_test_evaluation,
		#		train_seq_length,
		#		oov_token,
		#		NumberOfSamplesComputed,
		#		AttentionIndicator
		#		)				
		
		##Attention
		if AttentionIndicator:
			plotAttention(baseDirectory+'attentionSelf//train//',batch_iteration, tokenizer,
						modelPredictionText, modelPredictionAttention, 
						X_train_evaluation,
						NumberOfSamplesComputed,
						train_seq_length,
						oov_token)
		
			plotAttention(baseDirectory+'attentionSelf//validation//',batch_iteration, tokenizer,
						modelPredictionText, modelPredictionAttention, 
						X_val_evaluation,
						NumberOfSamplesComputed,
						train_seq_length,
						oov_token)
		
		
		
			plotAttention2(baseDirectory+'attention//train//',batch_iteration, tokenizer,
						modelTrainingText, modelTrainingAttention, 
						X_train_evaluation2,
						NumberOfSamplesComputed,
						train_seq_length,
						oov_token)
			plotAttention2(baseDirectory+'attention//validation//',batch_iteration, tokenizer,
						modelTrainingText, modelTrainingAttention, 
						X_val_evaluation2,
						NumberOfSamplesComputed,
						train_seq_length,
						oov_token)
			plotAttention2(baseDirectory+'attention//test//',batch_iteration, tokenizer,
						modelTrainingText, modelTrainingAttention, 
						X_test_evaluation2,
						NumberOfSamplesComputed,
						train_seq_length,
						oov_token)	
						
			getAttention(modelTrainingText,modelTrainingAttention,tokenizer,
				[categoryData[0][ValidationIndices],categoryData[1][ValidationIndices],categoryData[2][ValidationIndices],categoryData[3][ValidationIndices],sequenceData[ValidationIndices,0:train_seq_length]],
				sequenceData[ValidationIndices,1:train_seq_length+1],
				vocab_size,
				train_seq_length,
				baseDirectory,
				batch_iteration		
				)			
						
		
		#Train_Sentences = ReturnComputedSamples(tokenizer, 
		#										modelPredictionText, modelPredictionAttention, 
		#										X_train_evaluation,
		#										train_seq_length,
		#										oov_token)
		#Validation_Sentences = ReturnComputedSamples(tokenizer, 
		#										modelPredictionText, modelPredictionAttention, 
		#										X_val_evaluation,
		#										train_seq_length,
		#										oov_token)

		
		#Train_predictions = Classifier.predict(Train_Sentences)
		#Validation_predictions = Classifier.predict(Validation_Sentences)
		
		#counterVariable = 0
		#for category in category_names:
		
		#	if category != 'price':
		#		classifier_train_losses[counterVariable].append(K.mean(categorical_accuracy(Train_predictions[counterVariable],X_train_evaluation[counterVariable])))
		#		classifier_validation_losses[counterVariable].append(K.mean(categorical_accuracy(Validation_predictions[counterVariable],X_val_evaluation[counterVariable])))
		#		PlotLossesWithLine(baseDirectory+'ClassifierLoss//'+category+'//','accuracy',batch_iteration,classifier_train_losses[counterVariable],classifier_validation_losses[counterVariable],
		#		classifierLossValues[counterVariable]
		#		)
		#	else:
		#		classifier_train_losses[counterVariable].append(mean_squared_error(Train_predictions[counterVariable],X_train_evaluation[counterVariable]))
		#		classifier_validation_losses[counterVariable].append(mean_squared_error(Validation_predictions[counterVariable],X_val_evaluation[counterVariable]))
		#		PlotLossesWithLine(baseDirectory+'ClassifierLoss//'+category+'//','mse',batch_iteration,classifier_train_losses[counterVariable],classifier_validation_losses[counterVariable],
		#		classifierLossValues[counterVariable]
		#		)
	
		#	counterVariable+=1
		
	
	
	
	

#print(getTrainingBatch(TrainingIndices,4))	













	
	

	
	
	
	



