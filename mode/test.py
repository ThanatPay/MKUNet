import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import cv2
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from utils.processing import create_dir,save_results
from utils.metric import recall_m,jacard_coef_loss,iou_loss,focal_loss,ssim_loss,iou,dice_coef,dice_loss
from voli.UNet import UNet

H = 512
W = 512
input_shape = (512, 512, 3)
OUTPUT_CHANNELS = 1
""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Directory for storing files """
create_dir("results")

# """ Loading model """
model = UNet(input_shape, OUTPUT_CHANNELS)
model.load_weights("model/model_unet.h5")

""" Load the dataset """
test_x = sorted(glob(os.path.join("data", "test", "image", "*")))
test_y = sorted(glob(os.path.join("data", "test", "mask", "*")))
print(f"Test: {len(test_x)} - {len(test_y)}")

""" Evaluation and Prediction """
SCORE = []
for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
    """ Extract the name """
    name = x.split("/")[-1].split(".")[0]

    """ Reading the image """
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    x = image/255.0
    x = np.expand_dims(x, axis=0)

    """ Reading the mask """
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = mask/255.0
    y = y > 0.5
    y = y.astype(np.int32)

    """ Prediction """
    y_pred = model.predict(x)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)

    """ Saving the prediction """
    save_image_path = f"results/{name}.png"
    save_results(image, mask, y_pred, save_image_path)

    """ Flatten the array """
    y = y.flatten()
    y_pred = y_pred.flatten()
    if (name[0] == 'B'):
        y = (y+1)%2
        y_pred = (y_pred+1)%2

    """ Calculating the metrics values """
    acc_value = accuracy_score(y, y_pred)
    f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
    jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
    recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
    precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
    percentage = sum(y_pred)/(512*512)
    SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value,percentage])

""" Metrics values """
score = [s[1:]for s in SCORE]
score = np.mean(score, axis=0)
print(f"Accuracy: {score[0]:0.5f}")
print(f"F1: {score[1]:0.5f}")
print(f"Jaccard: {score[2]:0.5f}")
print(f"Recall: {score[3]:0.5f}")
print(f"Precision: {score[4]:0.5f}")

df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision","percentage"])
df.to_csv("score/unet.csv")
