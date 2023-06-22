import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Path of the file to read

landmark_file_df = "../GoogleISLDatasetBatched"
train_df = "../train.csv"

# # print the list of columns in the dataset to find the name of the prediction target
# sample = pd.read_parquet("../train_landmark_files/16069/100015657.parquet")
# print(sample.head())

# # pick the left hand and right hand points
# sample_left_hand = sample[sample.type == "left_hand"]
# sample_right_hand = sample[sample.type == "right_hand"]

# # display(sample_left_hand)

# # edges that represents the hand edges
# # How he knows the edges, so a mystery 
# edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(0,17),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
#          (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20)]

# # plotting a single frame into matplotlib
# def plot_frame(df, frame_id, ax):
#     df = df[df.frame == frame_id].sort_values(['landmark_index'])
#     x = list(df.x)
#     y = list(df.y)
    
#     # plotting the points
#     ax.scatter(df.x, df.y, color='dodgerblue')
#     for i in range(len(x)):
#         ax.text(x[i], y[i], str(i))
    
#     # plotting the edges that represents the hand
#     for edge in edges:
#         ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color='salmon')
#         ax.set_xlabel(f"Frame no. {frame_id}")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])

# # plotting the multiple frames
# def plot_frame_seq(df, frame_range, n_frames):
#     frames = np.linspace(frame_range[0],frame_range[1],n_frames, dtype = int, endpoint=True)
#     fig, ax = plt.subplots(n_frames, 1, figsize=(5,25))
#     for i in range(n_frames):
#         plot_frame(df, frames[i], ax[i])
        
#     plt.show()

# plot_frame_seq(sample_left_hand, (178,186), 10)

# Set constants and pick important landmarks
LANDMARK_IDX = [0,9,11,13,14,17,117,118,119,199,346,347,348] + list(range(468,543))
DS_CARDINALITY = 185
VAL_SIZE  = 20
N_SIGNS = 250
ROWS_PER_FRAME = 543

def preprocess(ragged_batch, labels):
    ragged_batch = tf.gather(ragged_batch, LANDMARK_IDX, axis=2)
    ragged_batch = tf.where(tf.math.is_nan(ragged_batch), tf.zeros_like(ragged_batch), ragged_batch)
    return tf.concat([ragged_batch[...,i] for i in range(3)],-1), labels

dataset = tf.data.Dataset.load(landmark_file_df)
dataset = dataset.map(preprocess)
val_ds = dataset.take(VAL_SIZE).cache().prefetch(tf.data.AUTOTUNE)
train_ds = dataset.skip(VAL_SIZE).cache().shuffle(20).prefetch(tf.data.AUTOTUNE)

# include early stopping and reducelr
def get_callbacks():
    return [
            tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor = "val_accuracy",
            factor = 0.5,
            patience = 3
        ),
    ]

# a single dense block followed by a normalization block and relu activation
def dense_block(units, name):
    fc = tf.keras.layers.Dense(units)
    norm = tf.keras.layers.LayerNormalization()
    act = tf.keras.layers.Activation("gelu")
    return lambda x: act(norm(fc(x)))
def dense_b1(units, name):
    fc = tf.keras.layers.Dense(units,activation="gelu")
    return lambda x: fc(x)
def dense_b(units, name):
    fc = tf.keras.layers.Dense(units,activation="softmax")
    return lambda x: fc(x)
def classifier1(lstm_units):
#     lstm = tf.keras.layers.LSTM(lstm_units)
    lstm = tf.keras.layers.LSTM(lstm_units,return_sequences=True)
#     norm = tf.keras.layers.LayerNormalization()
#     act = tf.keras.layers.Activation("gelu")
#     out = tf.keras.layers.Dense(N_SIGNS, activation="softmax")
#     lstm = tf.keras.layers.LSTM(int(lstm_units/2))
#     out = tf.keras.layers.Dense(N_SIGNS, activation="softmax")
    return lambda x: lstm(x)
# the lstm block with the final dense block for the classification
def classifier(lstm_units):
#     lstm = tf.keras.layers.LSTM(lstm_units)
    lstm = tf.keras.layers.LSTM(lstm_units)
#     norm = tf.keras.layers.LayerNormalization()
#     act = tf.keras.layers.Activation("gelu")
#     out = tf.keras.layers.Dense(N_SIGNS, activation="softmax")
#     lstm = tf.keras.layers.LSTM(int(lstm_units/2))
#     out = tf.keras.layers.Dense(N_SIGNS, activation="softmax")
    return lambda x: lstm(x)
# choose the number of nodes per layer
encoder_units = [512,256] # tune this
lstm_units = 256 # tune this

#define the inputs (ragged batches of time series of landmark coordinates)
inputs = tf.keras.Input(shape=(None,3*len(LANDMARK_IDX)), ragged=True)

# dense encoder model
x = inputs
for i, n in enumerate(encoder_units):
    print(n)
    x = dense_block(n, f"encoder_{i}")(x)
# x= dense_b1(256,"encoder250")(x)    
x = tf.keras.layers.Dropout(0.4)(x)

# classifier model
x = classifier1(lstm_units)(x)
# print(x)

# x = tf.expand_dims(out, axis=0)
# x = classifier(lstm_units,"LSTMx")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = classifier(lstm_units)(x)
x = tf.keras.layers.Dropout(0.4)(x)
x= dense_b1(256,"encoder250")(x)
out = dense_b(250,"encoder250")(x)
# tensor_2 = tf.expand_dims(out, axis=1)
# tensor_2 = tf.expand_dims(tensor_2, axis=1)

# Tile the tensor along the new dimensions to get shape (None, None, 256)
# tensor_3 = tf.tile(tensor_2, multiples=[1, tf.shape(out)[1], 1, 1])
# print(tensor_3)
model = tf.keras.Model(inputs=inputs, outputs=out)
model.summary()

# add a decreasing learning rate scheduler to help convergence
steps_per_epoch = DS_CARDINALITY - VAL_SIZE
boundaries = [steps_per_epoch * n for n in [30,50,70]]
values = [1e-3,1e-4,1e-5,1e-6]
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer = tf.keras.optimizers.Adam(lr_sched)

model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy","sparse_top_k_categorical_accuracy"])

# fit the model with 100 epochs iteration
model.fit(train_ds,
          validation_data = val_ds,
          callbacks = get_callbacks(),
          epochs = 150)

model.save('lstm.h5')

model.summary(expand_nested=True)

def get_inference_model(model):
    inputs = tf.keras.Input(shape=(ROWS_PER_FRAME,3), name="inputs")
    
    # drop most of the face mesh
    x = tf.gather(inputs, LANDMARK_IDX, axis=1)

    # fill nan
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    # flatten landmark xyz coordinates ()
    x = tf.concat([x[...,i] for i in range(3)], -1)

    x = tf.expand_dims(x,0)
    
    # call trained model
    out = model(x)
    
    # explicitly name the final (identity) layer for the submission format
    out = tf.keras.layers.Activation("linear", name="outputs")(out)
    
    inference_model = tf.keras.Model(inputs=inputs, outputs=out)
    inference_model.compile(loss="sparse_categorical_crossentropy",
                            metrics="accuracy")
    return inference_model


inference_model = get_inference_model(model)
inference_model.summary(expand_nested=True)

# save the model
converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
tflite_model = converter.convert()
model_path = "model.tflite"