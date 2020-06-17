import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import argparse

from Utils.data_loader import sort_filanames, image_loder
from Utils.preprocessing import preprocessing
from Models.Inceptionv3Tnsf import model

# construct the argument parser
ap = argparse.ArgumentParser()
# train.csv
ap.add_argument('-r', '--train', required=True,
                help='Path to train df')
# test.csv
ap.add_argument('-e', '--test', required=True,
                help='Path to train df')
# train dir
ap.add_argument('-d', '--train_dir', required=True,
                help='Path to train Directory')
ap.add_argument('-t', '--test_dir', required=True,
                help='Path to test Directory')
# submission dir
ap.add_argument('-s', '--subDir', required=True,
                help='Path to submission file')
ap.add_argument('-o', '--output', required=True,
                help='Path to save the plot')
args = vars(ap.parse_args())

# load csv files
traindf = pd.read_csv(args['train'])
testdf = pd.read_csv(args['test'])

# sort the image indexes for image loading
traindfSorted = sort_filanames(traindf)
testdfSorted = sort_filanames(testdf)

# load the images from respective directory
trainImages = image_loder(args['train_dir'], traindfSorted)
testImages = image_loder(args['test_dir'], testdfSorted)

# resize the images to approximately mean shape of 440*440*3
trainProcessed = preprocessing(trainImages)
testProcessed = preprocessing(testImages)

# convert them to numpy arrays
Features = np.array(trainProcessed)
X_test = np.array(testProcessed)

# normalize them by dividing 255
Features = Features / 255.0
X_test = X_test / 255.0

# let's play with our target column
# convert them to Label Encoding so that we can later apply Inverse label encoding
le = LabelEncoder()
traindfSorted['labels'] = le.fit_transform(traindfSorted['target'])
labels = to_categorical(traindfSorted['labels'], num_classes=8)

print('Shape of label vector :', labels.shape)

X_train, X_val, y_train, y_val = train_test_split(Features, labels,
                                                  test_size=0.15, random_state=42)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

model = model()

H = model.fit(X_train, y_train,
              batch_size=1, epochs=20,
              validation_data=(X_val, y_val),
              shuffle=True, verbose=0,
              steps_per_epoch=len(X_train)//8,
              validation_steps=len(X_val)//4)

predictions = model.predict(X_test, batch_size=1).argmax(axis=1)
predictionTarget = le.inverse_transform(predictions)
submission = pd.DataFrame()
submission['Image'] = testdfSorted['Image']
submission['target'] = predictionTarget
submission['Image'] = submission['Image'].apply(lambda x: str(x) + '.jpg')
submission.to_csv(args['subDir'], index=False)

plt.figure()
plt.plot(np.arange(0, 20), H.history["accuracy"], label="training_accuracy")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and accuracy")
plt.xlabel("#Epochs")
plt.ylabel("loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

# print(classification_report(le.fit_transform(traindfSorted['target']),
# model.predict_classes(Features, batch_size=1)))
