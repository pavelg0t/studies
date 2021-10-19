import tensorflow as tf
import tensorflow_addons as tfa
import random

def load_imgs(img_paths):
  #Load images
  x = tf.io.read_file(img_paths[0])
  x = tf.io.decode_png(x)
  y = tf.io.read_file(img_paths[1])
  y = tf.io.decode_png(y)
  return (x,y)

def preprocess(x,y):
  #Normalize
  x = tf.keras.layers.Rescaling(1/255)(x)
  y = tf.keras.layers.Rescaling(1/255)(y)
  y = tf.cast(y, tf.int32)
  # Add Pixel-wise loss weights
  ratio = ( (tf.size(y)) - tf.reduce_sum(y))/(tf.reduce_sum(y)+1) # ratio of areas
  weights = tf.keras.layers.Rescaling(ratio)(y)  # Add weights for vessels' pixels
  weights +=1
  return (x,y,weights)

def augment(x,y):
  # Rotate randomlly
  beta = random.randint(0, 360)/180*3.1415
  x = tfa.image.rotate(x,angles=beta)
  y = tfa.image.rotate(y,angles=beta)
  # Deform imgs
  # TO DO:
  return (x,y)


def generate_dataset(im_pairs,BATCH_SIZE,SHUFFLE_SIZE,split_ration):

  ds = tf.data.Dataset.from_tensor_slices(im_pairs)

  # Split dataset for training and validation
  ds_train = ds.take( int(len(im_pairs)*split_ration) )
  ds_val = ds.skip(int(len(im_pairs)*split_ration))

  ds_train = ds_train.map(load_imgs,num_parallel_calls=tf.data.AUTOTUNE).shuffle(SHUFFLE_SIZE)
  ds_train = ds_train.map(augment,num_parallel_calls=tf.data.AUTOTUNE)
  ds_train = ds_train.map(preprocess,num_parallel_calls=tf.data.AUTOTUNE)
  ds_train = ds_train.batch(BATCH_SIZE)
  ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


  ds_val = ds_val.map(load_imgs,num_parallel_calls=tf.data.AUTOTUNE).shuffle(SHUFFLE_SIZE)
  ds_val = ds_val.map(preprocess,num_parallel_calls=tf.data.AUTOTUNE)
  ds_val = ds_val.batch(BATCH_SIZE)
  ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

  return (ds_train, ds_val)