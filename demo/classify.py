#Imports
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

######### Helper Functions ##########
# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy',
	              optimizer='adam', metrics=['accuracy'])
	return model

# smaller model
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy',
	              optimizer='adam', metrics=['accuracy'])
	return model


# larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(30, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy',
	              optimizer='adam', metrics=['accuracy'])
	return model


def train_network_with(network_creator, model_type):
	"Trains and evaluates a neural network created using the network_creator function provided as input"

	print("Training a {} model with standardized data".format(model_type))
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasClassifier(
		build_fn=network_creator, epochs=100, batch_size=5, verbose=0)))
	pipeline = Pipeline(estimators)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
	print("Results: %.2f%% (%.2f%%)" %
            (results.mean()*100, results.std()*100))

########Main experiment initialization ###########


# fix random seed for reproducibility
#Need to add backslashes to escape the slashes in the file path for Windows
file_location = "C:\\Users\\cr89536\\Code\\ML\\demo\\datasets\\sonar.csv"
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv(file_location, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:60].astype(float) #all rows and columns up to column 60, exclusive
Y = dataset[:, 60] 

# encode class values as integers:
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#####Train and run models ######
print('Start the training and evaluation gauntlet!')

# train and evaluate the baseline model
estimator = KerasClassifier(
	build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results for baseline model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#Train and evaluate the baseline network with standardied data
train_network_with(create_baseline, 'baseline')

#Train and evaluate a smaller network with standardized data
train_network_with(create_smaller, 'smaller')

#Train and evaluate a larger network with standardized data
train_network_with(create_larger, 'larger')

print('All done!')
