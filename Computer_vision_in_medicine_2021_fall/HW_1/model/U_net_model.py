import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Input, Conv2D, Lambda, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization
from tensorflow.keras import Model


def get_model(WIDTH,HEIGHT,INPT_CHANNELS,N_ch=16):

  '''
          WIDTH: width of input image
         HEIGHT: height of input image 
  INPT_CHANNELS: number of color chanels in the input image (3 for RGB image)
           N_ch: number of channels/filters in the Conv2D layers of block 'c1'
  '''
  # Model input
  input = Input(shape=(WIDTH,HEIGHT,INPT_CHANNELS), name='Input')
  

  #Contraction path (Encoder) 
  c1 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(input)
  c1 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c1)
  c1 = BatchNormalization()(c1)  
  c1 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c1)
  c1 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c1)
  p1 = MaxPool2D(pool_size=(2, 2))(c1)

  c2 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(p1)
  c2 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c2)
  c2 = BatchNormalization()(c2)  
  c2 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c2)
  c2 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c2)
  p2 = MaxPool2D(pool_size=(2, 2))(c2)

  c3 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(p2)
  c3 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c3)
  c3 = BatchNormalization()(c3) 
  c3 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c3)
  c3 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c3)
  p3 = MaxPool2D(pool_size=(2, 2))(c3)

  c4 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(p3)
  c4 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c4)
  c4 = BatchNormalization()(c4) 
  c4 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c4)
  c4 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c4)
  p4 = MaxPool2D(pool_size=(2, 2))(c4)

  c5 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(p4)
  c5 = Conv2D(filters=N_ch*16, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c5)
  c5 = BatchNormalization()(c5)  
  c5 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c5)
  c5 = Conv2D(filters=N_ch*16, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c5)


  #Expansion path (Decoder)
  u6 = Conv2DTranspose(filters=N_ch*8, kernel_size=(2,2), strides=(2, 2), padding="same")(c5)
  u6 = Concatenate()([u6,c4])
  c6 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(u6)
  c6 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c6)
  c6 = BatchNormalization()(c6)  
  c6 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c6)
  c6 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c6)

  u7 = Conv2DTranspose(filters=N_ch*4, kernel_size=(2,2), strides=(2, 2), padding="same")(c6)
  u7 = Concatenate()([u7,c3])
  c7 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(u7)
  c7 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c7)
  c7 = BatchNormalization()(c7)  
  c7 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c7)
  c7 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c7)

  u8 = Conv2DTranspose(filters=N_ch*2, kernel_size=(2,2), strides=(2, 2), padding="same")(c7)
  u8 = Concatenate()([u8,c2])
  c8 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(u8)
  c8 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c8)
  c8 = BatchNormalization()(c8)  
  c8 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c8)
  c8 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c8)

  u9 = Conv2DTranspose(filters=N_ch, kernel_size=(2,2), strides=(2, 2), padding="same")(c8)
  u9 = Concatenate()([u9,c1])
  c9 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(u9)
  c9 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c9)
  c9 = BatchNormalization()(c9)  
  c9 = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0] ], 'REFLECT'))(c9)
  c9 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c9)

  output = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid',padding="same")(c9)

  model = Model(inputs=input, outputs=output)

  return model
