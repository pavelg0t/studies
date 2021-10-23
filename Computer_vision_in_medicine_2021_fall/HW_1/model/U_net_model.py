import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Input, Conv2D, Lambda, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization
from tensorflow.keras import Model
import math

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



# This layer could be used in the model architecture directly (but with additional computation cost)
# Extract image patches for inferance
class CreatePatches( tf.keras.layers.Layer ):

  def __init__( self , patch_size ):
    super( CreatePatches , self ).__init__()
    self.patch_size = patch_size

  def call(self, inputs ):
    # If width or/and height of the rectangular image 
    # are not multiples of 'patch_size' => then pad image with zeros
    # print('Entering "Call"')
    h,w = inputs.shape[-3:-1]
    w_new = self.patch_size*((w//self.patch_size)+1) if (w - self.patch_size*(w//self.patch_size))>0 else w
    h_new = self.patch_size*((h//self.patch_size)+1) if (h - self.patch_size*(h//self.patch_size))>0 else h
    # Pad with Reflection
    paddings = tf.constant([[0, h_new-h,], [0, w_new-w],[0,0]]) # [[Up,Down],[Left,Right],[channels]]
    inputs = tf.pad(inputs, paddings, "REFLECT") 
    # Normalize input
    inputs = tf.keras.layers.Rescaling(1/255)(inputs)
    # Initialize a first patch (top-left corner of the image)
    patches = tf.expand_dims(inputs[0:self.patch_size, 0:self.patch_size, : ],axis=0)
    for i in range( 0 , h_new , self.patch_size ):
        for j in range( 0 , w_new , self.patch_size ):
            if (i!=0) | (j!=0):
              patch = tf.expand_dims(inputs[ i : i + self.patch_size, j : j + self.patch_size , : ],axis=0)
              patches = tf.concat( [patches,patch], axis=0 )
    return patches



def concat_to_vector(x,ax):
  res = x[0,:,:,:]
  for i in range(1,x.shape[0]):
    res = tf.concat([res,x[i,:,:,:]],axis=ax)
  return res


def stich_patches(patches,h,w):
  '''
  patches: Tensor of shape [Np,L,L,C]
        h: height of the final image (multiples of L)
        w: width of the final image(multiples of L)
  '''
  L = patches.shape[-2] #side-lenght of the patches
  img = concat_to_vector(patches[:w//L,:,:,:],ax=1)
  for i in range(1,h//L):
    img = tf.concat([img,concat_to_vector(patches[i*w//L:(i+1)*w//L,:,:,:],ax=1)],axis=0)
  return img


def predict(model, image, patch_size):
  '''
  model: tf model 
  image: Tensor type image
  patch_size: size of a square patch the image will be cropped and stacked into a Tensor before prediction
  '''
  patches = CreatePatches( patch_size=patch_size )(image)
  preds = model.predict(patches)
  h_pad =  math.ceil(image.shape[-3]/patch_size)*patch_size
  w_pad =  math.ceil(image.shape[-2]/patch_size)*patch_size 
  prediction = stich_patches(preds,h_pad,w_pad)
  return prediction[:image.shape[-3],:image.shape[-2],:]
