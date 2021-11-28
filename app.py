# Import useful libraries
import pandas as pd
import json
import numpy as np
from scipy.stats.stats import mode
import seaborn as sns
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize, StandardScaler

# Classifier models
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, GridSearchCV 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report


# Statistic libraries
from scipy.optimize import fmin, minimize_scalar
from scipy import stats

# Importing Keras library from Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from keras.models import load_model
from os import makedirs



def show_bands(row):
  print(f'Index : {row.name}')
  f, (img1, img2) = plt.subplots(1, 2, figsize = (10,10))
  img1.imshow(np.array(row.band_1).reshape(75,75))
  img1.set_title("band_1")
  img2.imshow(np.array(row.band_2).reshape(75,75))
  img2.set_title("band_2")
  img1.set_yticks([])
  img1.set_xticks([])
  img2.set_yticks([])
  img2.set_xticks([])
  plt.show()

class EvaluateAndReport:
  def __init__(self, df, X_train, X_test, y_train, y_test, scoring = 'f1', grid_cv = 5, cv_cv = 10, best_scores_t = {}):
    self.df = df
    self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
    self.scoring = scoring
    self.grid_cv = grid_cv
    self.cv_cv = cv_cv
    #self.threshold = threshold
    self.best_scores_t = best_scores_t


  def grid_report(self, classifier = None, param_grid = None):
    if classifier is None:
      classifier = self.last_best
    if not(param_grid is None):
      print('--- Grid Search Report ---')
      grid_search = GridSearchCV(classifier, param_grid, cv = self.grid_cv, scoring = self.scoring, return_train_score = True)
      grid_search.fit(self.X_train, self.y_train)

      print(f'Best parameters for {classifier} :\n {grid_search.best_params_} with {self.scoring} score {round(grid_search.best_score_,3)}')

      self.last_best = grid_search.best_estimator_
    else:
      print("cross_report should be used when there is no param_grid")


  def cross_report(self, classifier = None):
    if classifier is None:
      classifier = self.last_best
    print('--- Cross Validation Report ---')
    scorings = ('accuracy', 'f1', 'precision', 'recall')
    scores = cross_validate(classifier, self.X_train, self.y_train, cv = self.cv_cv, scoring = scorings)
    self.best_scores_t = {scoring : round(np.mean(scores["test_"+scoring]),3) for scoring in scorings}
    [print(f'Mean {scoring} score {round(np.mean(scores["test_"+scoring]),3)} with std {round(np.std(scores["test_"+scoring]),3)}') for scoring in scorings]


  def test_report(self, classifier = None):
    if classifier is None:
      classifier = self.last_best
    print('--- Test Sample Report ---')
    classifier.fit(self.X_train, self.y_train)
    self.last_y_pred = classifier.predict(self.X_test)

    print(f'Scores for the test sample :')
    print(f' accuracy : {round(accuracy_score(self.y_test, self.last_y_pred),3)}')
    print(f' f1 : {round(f1_score(self.y_test, self.last_y_pred),3)}')
    print(f' precision : {round(precision_score(self.y_test, self.last_y_pred),3)}')
    print(f' recall : {round(recall_score(self.y_test, self.last_y_pred),3)}')

    print('Confusion matrix for the test sample :')
    cm = confusion_matrix(self.y_test, self.last_y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Boat','Iceberg']).plot(cmap="Blues", values_format='')


  def grid_cross_test_report(self, classifier = None, param_grid = None):
    if classifier is None:
      classifier = self.last_best
    if not(param_grid is None):
      self.grid_report(classifier, param_grid)
      self.cross_report(self.last_best)
      self.test_report(self.last_best)
    else:
      print('cross_test_report should be used when there is no param_grid')

  def cross_test_report(self, classifier = None):
    if classifier is None:
      classifier = self.last_best
    self.cross_report(classifier)
    self.test_report(classifier)

  def cross_proba_report(self, classifier = None, threshold = 0.5):
    if classifier is None:
      classifier = self.last_best
    print(f'--- Cross Validation with {threshold} Threshold Report ---')
    y_train_scores = cross_val_predict(self.last_best, self.X_train, self.y_train, cv=self.cv_cv, method='predict_proba')
    y_train_scores = np.array([y[1] for y in y_train_scores])
    self.print_proba_scores(self.calc_proba_scores(y_train_scores, threshold))

  def calc_proba_scores(self, y_train_scores, threshold):
    y_pred = (y_train_scores >= threshold)
    return (accuracy_score(self.y_train, y_pred), f1_score(self.y_train, y_pred, zero_division=0), precision_score(self.y_train, y_pred, zero_division=0), recall_score(self.y_train, y_pred, zero_division=0))

  def print_proba_scores(self, scores):
    print(f' accuracy : {round(scores[0],3)}')
    print(f' f1 : {round(scores[1],3)}')
    print(f' precision : {round(scores[2],3)}')
    print(f' recall : {round(scores[3],3)}')

  def full_cross_proba_report(self, classifier = None, res = 50):
    if classifier is None:
      classifier = self.last_best
    print(f'--- Threshold Report ---')
    y_train_scores = cross_val_predict(self.last_best, self.X_train, self.y_train, cv=self.cv_cv, method='predict_proba')
    y_train_scores = np.array([y[1] for y in y_train_scores])
    probas = np.linspace(0.0,1.0,res)
    full_scores = np.array([self.calc_proba_scores(y_train_scores, threshold) for threshold in probas])

    plt.figure(figsize=(8,6))
    plt.plot(probas, full_scores[:,0], "b-", label="Accuracy", linewidth=2)
    plt.plot(probas, full_scores[:,1], "g-", label="F1", linewidth=2)
    plt.plot(probas, full_scores[:,2], "r-", label="Precision", linewidth=2)
    plt.plot(probas, full_scores[:,3], "c-", label="Recall", linewidth=2)
    plt.xlabel("Probablility threshold", fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.legend(loc="best", fontsize=16)
    plt.ylim([0.4, 1])
    plt.xlim([0, 1])
    plt.grid(b=True, linestyle='-')

    #Searching for optimals
    acc_opt = minimize_scalar(lambda x : -self.calc_proba_scores(y_train_scores, x)[0], bracket=(0.0,1.0))
    if acc_opt.x < 0: acc_opt.x = 0.0
    elif acc_opt.x > 1 : acc_opt.x = 1.0
    print(f'Optimal threshold for accuracy is {round(acc_opt.x,2)} with:')
    self.print_proba_scores(self.calc_proba_scores(y_train_scores, acc_opt.x))

    acc_opt = minimize_scalar(lambda x : -self.calc_proba_scores(y_train_scores, x)[1], bracket=(0.0,1.0))
    if acc_opt.x < 0: acc_opt.x = 0.0
    elif acc_opt.x > 1 : acc_opt.x = 1.0
    print(f'Optimal threshold for f1 is {round(acc_opt.x,2)} with:')
    self.print_proba_scores(self.calc_proba_scores(y_train_scores, acc_opt.x))

  def plot_precision_recall(self, classifier = None):
    if classifier is None:
      classifier = self.last_best
    print('--- Precision Recall Curves ---')
    y_train_scores = cross_val_predict(self.last_best, self.X_train, self.y_train, cv=self.cv_cv, method='predict_proba')
    y_train_scores = np.array([y[1] for y in y_train_scores])
    precisions, recalls, probas = precision_recall_curve(self.y_train, y_train_scores)

    plt.figure(0)
    plt.plot(probas, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(probas, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Probablility threshold", fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.legend(loc="best", fontsize=16)
    plt.ylim([0, 1])

    plt.figure(1)
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

  def plot_mistakes(self, nb_samples = 1, nature = None, random_state = None):
    #Plots bands for wrongly predicted individuals
    #By default, false positives and false negatives are shown. If one is chosen in 'nature', the other ones won't be shown
    df_res = self.df.join(pd.DataFrame(data=self.last_y_pred, index=self.y_test.index, columns=['is_iceberg_pred']))
    
    if nature != 'boat_as_iceberg':
      print('--- Icebergs predicted as boats ---')
      df_res[(df_res['is_iceberg_pred'] == 0) & df_res['is_iceberg'] == 1].sample(nb_samples, random_state = random_state).apply(show_bands, axis=1)
    if nature != 'iceberg_as_boat':
      print('--- Boats predicted as Icebergs ---')
      df_res[(df_res['is_iceberg_pred'] == 1) & df_res['is_iceberg'] == 0].sample(nb_samples, random_state = random_state).apply(show_bands, axis=1)

  def plot_predicted(self, nb_samples = 1, nature = None, random_state = None):
    #Plots bands for correclty predicted individuals
    #By default, correctly predicted icebergs and boats are shown. If one is chosen in 'nature', the others ones won't be shown
    df_res = self.df.join(pd.DataFrame(data=self.last_y_pred, index=self.y_test.index, columns=['is_iceberg_pred']))
    
    if nature != 'boats':
      print('--- Icebergs predicted correctly ---')
      df_res[(df_res['is_iceberg_pred'] == 1) & df_res['is_iceberg'] == 1].sample(nb_samples, random_state = random_state).apply(show_bands, axis=1)
    if nature != 'icebergs':
      print('--- Boats predicted correctly ---')
      df_res[(df_res['is_iceberg_pred'] == 0) & df_res['is_iceberg'] == 0].sample(nb_samples, random_state = random_state).apply(show_bands, axis=1)


# Adding more features to the initial dataset
def add_features(data, label) :
  data['max_b'+str(label)] = [np.max(np.array(matx)) for matx in data['band_'+str(label)]]
  data['min_b'+str(label)] = [np.min(np.array(matx)) for matx in data['band_'+str(label)]]
  data['median_b'+str(label)] = [np.median(np.array(matx)) for matx in data['band_'+str(label)]]
  data['mean_b'+str(label)] = [np.mean(np.array(matx)) for matx in data['band_'+str(label)]]
  data['argmax_b'+str(label)] = [np.argmax(np.array(matx)) for matx in data['band_'+str(label)]]
  data['argmin_b'+str(label)] = [np.argmin(np.array(matx)) for matx in data['band_'+str(label)]]
  data['band_'+str(label)] = [np.array(matx).reshape(75,75) for matx in data['band_'+str(label)]]
  return data

# Plotting the new feature in a histogram
def plot_features(data, name):
  plt.hist(data.loc[data.is_iceberg==1,name], bins=50, color='navy',alpha=0.5,label='Iceberg')
  plt.hist(data.loc[data.is_iceberg==0,name], bins=50,color='firebrick',alpha=0.5, label='Ship')
  plt.legend()
  plt.xlabel(name)
  plt.ylabel('Frequency')
  plt.show()

# Applying a PCA a returning a df with the specified number of principle components
def get_pca_df(df, X, pcs):
  X_std = StandardScaler().fit_transform(X)
  pca = PCA(pcs).fit(X_std)
  res = pd.DataFrame(pca.transform(X_std),columns=['PC%s' % _ for _ in range(pcs)], index=df.index)
  res = res.join(df['is_iceberg'])
  res = res.dropna()
  return res

def to_RGB(matx):
  normalized = (matx-np.min(matx))/(np.max(matx)-np.min(matx))
  img = Image.fromarray(plt.cm.jet(normalized, bytes=True))
  img = img.resize((300, 300), Image.ANTIALIAS)
  return (img)

def get_distrib(matx, display = True):
   # getting the elements as a 1D array
  data = matx.ravel()
  if display : 
    ## visual part of the function
    fig = plt.figure(figsize=(14,10))
    # showing the original image
    img = to_RGB(matx)
    ax = fig.add_subplot(1,2,1)
    ax.imshow(img)
    ax = fig.add_subplot(1,2,2)
    ax.hist(data, bins = 200, color ='blue')
    ax.set_xlabel("dB")
    ax.set_ylabel("Frequence")
    plt.show()
  return stats.describe(data)

class DeepLearningExplore:

  def __init__(self, df, X_train, X_test, y_train, y_test, best_model_=None, compiled_model=None, best_scores_t = {}):
    self.df = df
    self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
    self.best_scores_t = best_scores_t
    self.best_model = best_model_
    self.compiled_model = compiled_model

  def create_model(self, n_cv2D, n_dense, drop_cv2D=True, normalization =True, act_fun='relu', disp=True):
    model = keras.Sequential()
    input_shape_=(75, 75, 2)
    # Creating the minimum blocks needed and introducing data augmentation
    # Pretraitement, 'data augmentation'
    model.add(
      preprocessing.RandomFlip('horizontal') # flip gauche-à-droite
    ) 
    
    model.add(
      preprocessing.RandomFlip('vertical') # flip haut-en-bas
    )
    
    ########################
    ## CONVOLUTIONAL BASE ##
    ########################

    # Premier block avec conv2D et MaxPooling
    model.add(
      layers.Conv2D(filters=32, kernel_size=5, activation=act_fun, padding='valid')
    )
    model.add(
      layers.MaxPool2D()
    )

    # Ajout des blocks de convolution 2D
    for i in range(n_cv2D+1):
      model.add(
        layers.Conv2D(filters=32*(i+1), kernel_size=3, activation=act_fun, padding='valid')
      )
      model.add(
        layers.MaxPool2D()
      )
      model.add(
         layers.Dropout(0.2*drop_cv2D/(i+1))
      )

    # Ajout d'un flatten layer pour passer en couches denses
    model.add(layers.Flatten())

    ########################
    ##    DENSE HEAD      ##
    ########################
    for j in range(n_dense+1):
      model.add(
        layers.Dense(n_cv2D*32/(j+2), activation=act_fun)
      )
    if normalization:
      model.add(layers.BatchNormalization())

    # Ajout d'un couche sigmoid pour la classification
    model.add(layers.Dense(2, activation="softmax"))

    # Ajout d'un optimiseur
    model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
    )

    model.build(input_shape_)
    
    self.compiled_model = model

    # Affichage de la structure du réseau ainsi construit
    if disp:
      tf.keras.utils.plot_model(model)
      print(model.summary())
    self.best_model = model

    return model
    
  def get_best_trained(self, model, n_epoch, verbose = True):
    checkpointer = ModelCheckpoint(
      filepath="best_weights.hdf5", 
      monitor = 'val_accuracy',
      verbose=1, 
      save_best_only=True
    )
    history = model.fit(
      self.X_train,
      self.y_train,
      validation_data=(self.X_test, self.y_test),
      batch_size = 25,
      epochs = n_epoch,
      callbacks=[checkpointer],
    )
    self.best_model = model.load_weights('best_weights.hdf5')

    if verbose:
      history_df = pd.DataFrame(history.history)
      history_df.loc[0:, ['loss', 'val_loss']].plot()
      print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))
      history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
      print(("Maximum Validation Accuracy: {:0.4f}").format(history_df['val_binary_accuracy'].max()))
      

  def get_other_metrics(self, model=None):
    if model is None:
      model = self.best_model
    # getting the f1, recall and precision metrics
    y_pred = model.predict(self.X_test, batch_size=25, verbose =0)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(self.y_test, y_pred_bool))

    return classification_report(self.y_test, y_pred_bool)
  
  def transferred_learning():
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(75, 75, 2))

    # Adding top layers for classification
    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = Dense(1024, activation='relu')(out)

    # Using softmax activation for having the same predict_proba returns as sklearn methods
    predictions = Dense(2, activation='softmax')(out)

    # Constructing our "fined" model
    fined_model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional ResNet50 layers
    for layer in base_model.layers:
      layer.trainable = False
  
    # compile the model (should be done *after* setting layers to non-trainable)
    fined_model.compile(
      optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
      loss='binary_crossentropy',
      metrics=['binary_accuracy']
    )

    return fined_model

  # load models from file
  def load_all_models(n_start, n_end):
    all_models = list()
    for epoch in range(n_start, n_end):
      # define filename for this ensemble
      filename = 'models/model_' + str(epoch) + '.h5'
      # load model from file
      model = load_model(filename)
      # add to list of members
      all_models.append(model)
      print('>>>>>>> loaded %s' % filename)
    return all_models
  
  # make an ensemble prediction for multi-class classification
  def ensemble_predictions(self, members):
    # make predictions
    yhats = [model.predict(self.X_test) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    return result

  def evaluate_n_members(self, members, n_members):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = self.ensemble_predictions(subset, self.X_test)
    # calculate accuracy
    return accuracy_score(self.y_test, yhat)

  def horizontal_voting(self, n_epoch, n_save_after):
    makedirs('models')
    for i in range(n_epoch):
      #fit the model for a single epoch
      self.compiled_model.fit(self.X_train, self.y_train, epochs=1, verbose=0)
      #check if we should save the model
      if i >= n_save_after:
        self.compiled_model.save('models/model_' + str(i) + '.h5')
    members = list(reversed(self.load_all_models(n_save_after, n_epoch)))
    # evaluate different numbers of ensembles on hold out set
    single_scores, ensemble_scores = list(), list()
    for i in range(1, len(members)+1):
      # evaluate model with i members
      ensemble_score = self.evaluate_n_members(self, members, i)
      # evaluate the i'th model standalone
      _, single_score = members[i-1].evaluate(self.X_test, self.y_test, verbose=0)
      # summarize this step
      print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
      ensemble_scores.append(ensemble_score)
      single_scores.append(single_score)
    # summarize average accuracy of a single final model
    print('Accuracy %.3f (%.3f)' % (np.mean(single_scores), np.std(single_scores)))
    # plot score vs number of ensemble members
    x_axis = [i for i in range(1, len(members)+1)]
    plt.plot(x_axis, single_scores, marker='o', linestyle='None')
    plt.plot(x_axis, ensemble_scores, marker='o')
    plt.show()