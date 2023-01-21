import tensorflow as tf
from keras import layers

# A utility method to create a tf.data dataset from a Pandas DataFrame
def df_to_dataset(x, y):
  df = {key: value[:,tf.newaxis] for key, value in x.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), y))
  return ds

def add_cat_encoding_layer(name, ds, dtype, max_tokens = None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)
  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = ds.map(lambda x, y: x[name])
  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)
  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))

all_inputs = []
encoded_features = []

def add_crossed_layers(df, ds):
  c1 = tf.constant(df['sex'])
  c2 = tf.constant(df['pregnant'])
  layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(num_bins = 2, output_mode = 'one_hot')
  #np_layer = tf.make_ndarray(layer)
  #keras_layer = tf.keras.Input(shape=(1,), name='sex_cross_pregnancy')
  ds.append(layer((c1, c2)))
  print(ds) #TODO: try convert to nparray > add to df > convert df to ds > add keras layer as numerical layer?

def encode_cat_layers(catcols, ds):
  for header in catcols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = add_cat_encoding_layer(name=header, ds = ds, dtype='string', max_tokens=2)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

def encode_num_layers(numcols):
  for header in numcols:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    all_inputs.append(numeric_col)

def get_layers():
  return (all_inputs, encoded_features)

