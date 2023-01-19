import tensorflow as tf

# A utility method to create a tf.data dataset from a Pandas DataFrame
def df_to_dataset(x, y):
  ds = tf.data.Dataset.from_tensor_slices((dict(x), y))
  return ds


