# imports libraries / kütüphanelerin içe aktarılması
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import scikitplot as skplt 
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# load the user configs file which is present in config folder / Yapılandırma klasöründe bulunan kullanıcı yapılandırma dosyalarının yüklenmesi
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables / yapılandırma değişkenlerinin tanımlanması
test_size 		= config["test_size"]
seed 			= config["seed"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
results 		= config["results"]
classifier_path = config["classifier_path"]
train_path 		= config["train_path"]
num_classes 	= config["num_classes"]
classifier_path = config["classifier_path"]

# import features and labels / özelliklerin ve etiketlerin içe aktarılması
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels / özelliklerin ve etiketlerin biçimlerinin doğrulanması
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))
print ("[INFO] training started...")
# split the training and testing data / eğitim ve test verilerini ayırma işlemi
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model / model olarak lojistik regresyon kullanımı
print ("[INFO] creating model...")
model = LogisticRegression(random_state=seed)
model.fit(trainData, trainLabels)

# use rank-1 and rank-5 predictions / 1. ve 5. sıra tahminlerini kullanımı
print ("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
	# predict the probability of each class label and
	# take the top-5 class labels
	predictions = model.predict_proba(np.atleast_2d(features))[0]
	predictions = np.argsort(predictions)[::-1][:5]

	# rank-1 prediction increment
	if label == predictions[0]:
		rank_1 += 1

	# rank-5 prediction increment
	if label in predictions:
		rank_5 += 1

# convert accuracies to percentages / Doğruluk oranların yüzdesel hale dönüştürülmesi
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file / Doğrulukların dosyaya yazılması
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data / test verilerinin modelini değerlendirmek
preds = model.predict(testData)

# write the classification report to file / Sınıflandırma sonuçlarının dosyaya yazımı
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file / Sınıflandırıcının dosyaya dökümü
print ("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))



import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# display the confusion matrix / Karışıklık matrisinin gösterimi
print ("[INFO] confusion matrix")

# get the list of training labels / eğitim etiketlerinin listesini alın
labels = sorted(list(os.listdir(train_path)))
class_names =['barbados_cherry', 'betel', 'caricature_plant','sweet_potato','vieux_garcon']
# plot the confusion matrix / Karışıklık matrisinin çizimi
#ax= plt.subplot()
cm=confusion_matrix(testLabels, preds)
#cm =confusion_matrix(testLabels, preds, labels=[0,1,2,3,4])
#plt.figure()
plot_confusion_matrix(cm, classes=class_names,normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()
#ax= plt.subplot()
#sns.heatmap(cm,
#            annot=True,
#            cmap="Set2")
#ax.xaxis.set_ticklabels(['New', 'OldBad', 'OldGood']); ax.yaxis.set_ticklabels(['New', 'OldBad', 'OldGood']);
#plt.show()
