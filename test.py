from tensorflow import keras as ks, constant

layer = ks.layers.experimental.preprocessing.HashedCrossing(
    num_bins=5, output_mode='one_hot')
feat1 = constant(['A', 'B', 'A', 'B', 'A'])
feat2 = constant([101, 101, 101, 102, 102])
print(feat1, feat2)
print(layer((feat1, feat2)))
