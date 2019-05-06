# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"colab_type": "text", "id": "view-in-github", "cell_type": "markdown"}
# <a href="https://colab.research.google.com/github/RamitPahwa/modelblending/blob/master/BLEND.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 421}, "colab_type": "code", "id": "5FAbcU6Y-QJW", "outputId": "8ff3e473-e97a-47c0-e66b-f68a8e4c63c4"}
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from gensim.models import word2vec
import numpy as np

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10
epsilon = 1.0e-9

#word2vec
def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    num_features    # Word vector dimensionality                      
    min_word_count  # Minimum word count                        
    context         # Context window size 
    """
    
    # Set values for various parameters
    num_workers = 2  # Number of threads to run in parallel
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model
    print('Training Word2Vec model...')
    sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
    embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                        size=num_features, min_count=min_word_count,
                                        window=context, sample=downsampling)

    # If we don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    embedding_model.init_sims(replace=True)

    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                              np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    return embedding_weights




# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters
min_word_count = 1
context = 10

imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000,start_char=None,
                                                              oov_char=None, index_from=None)      
x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

vocabulary = imdb.get_word_index()
vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
vocabulary_inv[0] = "<PAD/>"

embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)


#teacher-predictions
#teacher = 'lstm'
#def get_teacher_predictions(teacher):
#  if teacher == 'lstm':
#    return y_train




# + {"colab": {"base_uri": "https://localhost:8080/", "height": 1033}, "colab_type": "code", "id": "lvGRSsFS-WtW", "outputId": "79117be7-5d13-425a-dc04-da3bb1aa3067"}
from tensorflow.keras.layers import Concatenate
import keras.backend as K
from keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Bidirectional, LSTM
from keras.losses import binary_crossentropy as logloss

'''
model = Sequential()
model.add(Embedding(len(vocabulary_inv), embedding_dim, name="embedding"))
model.add(Bidirectional(LSTM(1)))
model.add(Dropout(dropout_prob[0]))
model.add(Dense(1, activation='sigmoid'))

weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])

model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=20,
          validation_data=(x_test, y_test), verbose=2)
'''

y_train = np.column_stack((y_train, y_train))
y_test = np.column_stack((y_test, y_test))



def custom_loss(y_true, y_pred):
    lambda_param = 0
    soft_label = logloss(y_true[:, 1:], y_pred[:,])
    hard_label = logloss(y_true[:, :1], y_pred[:,])
    loss = lambda_param*soft_label + (1-lambda_param)*hard_label
    return loss

input_shape = (sequence_length, )
model_input = Input(shape=input_shape)
z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)
z = Dropout(dropout_prob[0])(z)

conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]


z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)
#model_output = Concatenate([model_output, model_output])

model = Model(model_input, model_output)
model.compile(loss=custom_loss, optimizer="adam", metrics=["accuracy"])

weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])

model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=20,
          validation_data=(x_test, y_test), verbose=2)


# + {"colab": {"base_uri": "https://localhost:8080/", "height": 51}, "colab_type": "code", "id": "t63szV-_0KaM", "outputId": "959bf7db-d83e-4a55-95a0-3e77e52d5a89"}


# + {"colab": {}, "colab_type": "code", "id": "__yjxTe2OVBq"}

