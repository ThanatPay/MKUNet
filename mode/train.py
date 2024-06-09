import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from utils.processing import create_dir,shuffling,load_data,tf_dataset
from utils.metric import recall_m,jacard_coef_loss,iou_loss,focal_loss,ssim_loss,iou,dice_coef,dice_loss
from voli.UNet import UNet

input_shape = (512, 512, 3)
OUTPUT_CHANNELS = 1
model = UNet(input_shape, OUTPUT_CHANNELS)
metrics = ['accuracy',recall_m,jacard_coef_loss]
model.compile(Adam(learning_rate=0.00001), loss=dice_loss, metrics=metrics)
model.summary()

tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)

np.random.seed(42)
tf.random.set_seed(42)

""" Directory for storing files """
create_dir("files")

""" Hyperparameters """
batch_size = 2
lr = 1e-4
num_epochs = 17
model_path = "model/model_unet.h5"
csv_path = os.path.join("files", "data.csv")

""" Dataset """
dataset_path = os.path.join("data")
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "val")

train_x, train_y = load_data(train_path)
train_x, train_y = shuffling(train_x, train_y)
valid_x, valid_y = load_data(valid_path)

print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Val: {len(valid_x)} - {len(valid_y)}")

train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)


callbacks = [
    ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    CSVLogger(csv_path),
    TensorBoard(log_dir='logs')]

model.fit(train_dataset, epochs=num_epochs,validation_data=valid_dataset,batch_size=2,callbacks=callbacks)