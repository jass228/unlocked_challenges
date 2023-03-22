#Default
import os
import numpy as np
import pandas as pd

#ML/DL
import tensorflow as tf

#Images
import cv2 as cv

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#Others
import random

def load_images(folder):
  """
  Description:
  ------------
  Function that allowed us to create image data

  Input:
  ------------
  folder: the folder where the images are located

  output:
  ------------
  An array of all the images

  """
  data_img = []
  onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
  onlyfiles = sorted(onlyfiles)
  #print(len(onlyfiles))
  for i in onlyfiles:
    img = cv.imread(os.path.join(folder, i))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if img is not None:
      data_img.append(img)
  return data_img

def show_images(data, n):
  """
  Description:
  ------------
  Function to print n ramdom images from one of the folders

  Input:
  ------------
  data: List of the image dataset
  n: Number of the image we want to plot

  Output:
  ------------
  plot images
  """
  plt.figure(figsize=(20,20))
  for i in range(n):
    a = random.randint(0, len(data))
    ax=plt.subplot(1,n,i+1)
    ax.title.set_text(a)
    plt.imshow(data[a])

def show_accuracy_loss(history_model):
  """
  Description:
  ------------
  Function to plot Accurary and Loss

  Input:
  ------------
  history_model: model history that we train

  Output:
  ------------
  plot accuracy and loss
  """
  acc = history_model.history['accuracy']
  val_acc = history_model.history['val_accuracy']

  loss = history_model.history['loss']
  val_loss = history_model.history['val_loss']

  epochs_range = range(1, len(acc) + 1)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

def compute_acc_loss(model, data):
  """
  Description:
  ------------
  Function to compute Accurary and Loss on new data

  Input:
  ------------
  model: model that we train
  data: test dataset

  Output:
  ------------
  show loss and accuracy
  """
  loss, accuracy = model.evaluate(data)
  print("initial loss: {:.2f}".format(loss))
  print("initial accuracy: {:.2f}".format(accuracy))


def submissionFile(test_dir, model, img_height, img_width, class_names):
    """
    Description:
    ------------
    Function to predict a test directory for submit

    Input:
    ------------
    test_dir: test directory
    model: model to use for prediction
    img_height: image height size 
    img_width: image width size
    class_names: list of class name

    Output:
    ------------
    DataFrame of the prediction
    """
    submi_data = []
    file_names = os.listdir(test_dir)

    for idx, file_name in enumerate(sorted(file_names)):
        img = tf.keras.utils.load_img(test_dir+file_name, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        if class_names[np.argmax(score)] == 'la_eterna':
            data = {
                "image" : file_name,
                "la_eterna": np.round(100 * np.max(score),2),
                "other_flower" : np.round(100 * np.min(score),2)
            }
        else:
            data = {
                "image" : file_name,
                "la_eterna": np.round(100 * np.min(score),2),
                "other_flower" : np.round(100 * np.max(score),2)
            }
        submi_data.append(data)

        #print(f"{file_name} : {np.min(score)}")

        #print("{} most likely belongs to {} with a {:.2f} percent confidence.".format(file_name,class_names[np.argmax(score)], 100 * np.max(score)))
    df = pd.DataFrame(submi_data)

    return df