import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


N_HIDDEN_1 = 80
N_HIDDEN_2 = 80
N_HIDDEN_3 = 40
#n_cont_inputs = train[:, cont_feats_idx].shape[1]
n_classes = 2

REGULARIZER = 'l2'
ACTIVATION = 'sigmoid'

# Learning parameters
LEARNING_RATE = 0.0001
N_EPOCHS = 50
N_ITERATIONS = 400
BATCH_SIZE = 250


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('TARGET')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def set_feature_columns(cont_feat_lookup, cat_feat_lookup):
    feature_columns = []

    # add numeric columns
    for header in cont_feat_lookup.feature.values:
        feature_columns.append(tf.feature_column.numeric_column(header))

    # add categorical embedded columns
    for header in cat_feat_lookup.feature.unique():
        dim = len(train_df[header].unique())
        cat_column = tf.feature_column.categorical_column_with_vocabulary_list(
            header, train_df[header].unique()
        )
        feature_columns.append(tf.feature_column.embedding_column(cat_column, dimension=dim))

    feature_columns = list(set(feature_columns))
    return feature_columns

def create_model(feature_columns, ACTIVATION=ACTIVATION, REGULARIZER=REGULARIZER):
    # define layers
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    hidden1 = Dense(N_HIDDEN_1, activation=ACTIVATION, kernel_regularizer=REGULARIZER)
    batch_norm1 = BatchNormalization()
    dropout1 = Dropout(rate=0.4)
    hidden2 = Dense(N_HIDDEN_2, activation=ACTIVATION, kernel_regularizer=REGULARIZER)
    batch_norm2 = BatchNormalization()
    dropout2 = Dropout(rate=0.2)
    hidden3 = Dense(N_HIDDEN_3, activation=ACTIVATION)
    batch_norm3 = BatchNormalization()
    output = Dense(1, activation=ACTIVATION)


    # define model
    model = Sequential([
                        feature_layer,
                        hidden1,
                        batch_norm1,
                        dropout1,
                        hidden2,
                        batch_norm2,
                        dropout2,
                        hidden3,
                        batch_norm3,
                        output
    ])
    return model

def plot_metrics(history):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'accuracy', 'auc']
    plt.figure(figsize=(18,24))
    for n, metric in enumerate(metrics):
        name = metric.replace('_', ' ').capitalize()
        plt.subplot(3,1, n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0],
                 linestyle='--', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)

        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.4, 1])
        else:
            plt.ylim([0,1])
        plt.legend()

    plt.show()

def main():
    ## read in training data
    merged_df = pd.read_csv('home_loans_train.csv')

    if 'Unnamed: 0' in merged_df.columns.values:
        merged_df = merged_df.drop('Unnamed: 0', axis=1)

    labels_df = pd.read_csv('home_loans_target.csv')
    if 'Unnamed: 0' in labels_df.columns.values:
        labels_df = labels_df.drop('Unnamed: 0', axis=1)

    merged_df['TARGET'] = labels_df['1'].astype(int)
    merged_df.columns = pd.Series(merged_df.columns).apply(lambda x: x.replace(' ', '_'))

    cont_feat_lookup = pd.read_csv('cont_feat_lookup.csv')
    cat_feat_lookup = pd.read_csv('cat_feat_lookup.csv')

    # ensure all categorical features are of int type
    for c in cat_feat_lookup.feature.values:
        merged_df = merged_df.astype({c:int})

    # create train test split
    train, test = train_test_split(merged_df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    test.to_csv('./data/test/home_loans_test.csv', index=False)

    # convert dataframes to tensorflow datasets
    train_ds = df_to_dataset(train, batch_size=BATCH_SIZE)
    val_ds = df_to_dataset(val, batch_size=BATCH_SIZE)
    test_ds = df_to_dataset(test, batch_size=BATCH_SIZE)

    feature_columns = set_feature_columns(cont_feat_lookup, cat_feat_lookup)

    model = create_model(feature_columns)

    ## early stopping
    early_stop = EarlyStopping(monitor='val_loss',
                                min_delta=0.01,
                                patience=5, verbose=20, mode='auto')

    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=['accuracy',
                        tf.keras.metrics.AUC(name='auc')
                        ])

    

    ## set checkpoints
    checkpoint_path = './models/checkpoints/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # create callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=True,
                                verbose=1)

    history = model.fit(train_ds,
            validation_data=val_ds,
            callbacks=[cp_callback, early_stop],
            epochs=N_EPOCHS)

    print(model.summary())

    plot_metrics(history)

    # save model
    model.save('models/home_loans_nn')
