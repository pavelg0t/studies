from tensorflow.keras.layers import Input, Conv2D, Lambda, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization
from tensorflow.keras import Model



def get_model(WIDTH,HEIGHT,INPT_CHANNELS,N_ch=16):

  '''
          WIDTH: width of input image
         HEIGHT: height of input image 
  INPT_CHANNELS: number of color chanels in the input image (3 for RGB image)
           N_ch: number of channels/filters in the Conv2D layers of block 'c1'
  '''
  # Defining our model architecture
  input = Input(shape=(WIDTH,HEIGHT,INPT_CHANNELS), name='Input')

  #Contraction path (Encoder)  
  c1 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(input)
  c1 = BatchNormalization()(c1)  
  c1 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c1)
  p1 = MaxPool2D(pool_size=(2, 2))(c1)

  c2 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(p1)
  c2 = BatchNormalization()(c2)  
  c2 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c2)
  p2 = MaxPool2D(pool_size=(2, 2))(c2)

  c3 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(p2)
  c3 = BatchNormalization()(c3)  
  c3 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c3)
  p3 = MaxPool2D(pool_size=(2, 2))(c3)

  c4 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(p3)
  c4 = BatchNormalization()(c4)  
  c4 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c4)
  p4 = MaxPool2D(pool_size=(2, 2))(c4)

  c5 = Conv2D(filters=N_ch*16, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(p4)
  c5 = BatchNormalization()(c5)  
  c5 = Conv2D(filters=N_ch*16, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c5)

  #Expansion path (Decoder)
  u6 = Conv2DTranspose(filters=N_ch*8, kernel_size=(2,2), strides=(2, 2), padding="same")(c5)
  u6 = Concatenate()([u6,c4])
  c6 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(u6)
  c6 = BatchNormalization()(c6)  
  c6 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c6)

  u7 = Conv2DTranspose(filters=N_ch*4, kernel_size=(2,2), strides=(2, 2), padding="same")(c6)
  u7 = Concatenate()([u7,c3])
  c7 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(u7)
  c7 = BatchNormalization()(c7)  
  c7 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c7)

  u8 = Conv2DTranspose(filters=N_ch*2, kernel_size=(2,2), strides=(2, 2), padding="same")(c7)
  u8 = Concatenate()([u8,c2])
  c8 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(u8)
  c8 = BatchNormalization()(c8)  
  c8 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c8)

  u9 = Conv2DTranspose(filters=N_ch, kernel_size=(2,2), strides=(2, 2), padding="same")(c8)
  u9 = Concatenate()([u9,c1])
  c9 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(u9)
  c9 = BatchNormalization()(c9)  
  c9 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='same')(c9)

  output = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid',padding="same")(c9)

  model = Model(inputs=input, outputs=output)

  return model