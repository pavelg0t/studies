import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Input, Conv2D, Lambda, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization
from tensorflow.keras import Model


class ReflectionPadding2D(Layer):
    '''
    Adding reflected padding before Conv2D() layer
    '''
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
    
    # For purooses as stated in: https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
    # def get_config(self):

    #     config = super().get_config().copy()
    #     config.update({
    #         'padding': self.vocab_size,
    #         'input_spec': self.input_spec
    #     })
    #     return config
    
    
    # For purooses as stated in:  https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        config.update({"padding": self.padding, "input_spec": self.input_spec})
        return config
    
    @classmethod
    def from_config(cls, config):
      """Creates a layer from its config.
      This method is the reverse of `get_config`,
      capable of instantiating the same layer from the config
      dictionary. It does not handle layer connectivity
      (handled by Network), nor weights (handled by `set_weights`).
      Arguments:
          config: A Python dictionary, typically the
              output of get_config.
      Returns:
          A layer instance.
      """
      return cls(**config)
    

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
  c1 = ReflectionPadding2D(padding=(1,1))(input)
  c1 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c1)
  c1 = BatchNormalization()(c1)  
  c1 = ReflectionPadding2D(padding=(1,1))(c1)
  c1 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c1)
  p1 = MaxPool2D(pool_size=(2, 2))(c1)

  c2 = ReflectionPadding2D(padding=(1,1))(p1)
  c2 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c2)
  c2 = BatchNormalization()(c2)  
  c2 = ReflectionPadding2D(padding=(1,1))(c2)
  c2 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c2)
  p2 = MaxPool2D(pool_size=(2, 2))(c2)

  c3 = ReflectionPadding2D(padding=(1,1))(p2)
  c3 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c3)
  c3 = BatchNormalization()(c3) 
  c3 = ReflectionPadding2D(padding=(1,1))(c3)
  c3 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c3)
  p3 = MaxPool2D(pool_size=(2, 2))(c3)

  c4 = ReflectionPadding2D(padding=(1,1))(p3)
  c4 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c4)
  c4 = BatchNormalization()(c4) 
  c4 = ReflectionPadding2D(padding=(1,1))(c4)
  c4 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c4)
  p4 = MaxPool2D(pool_size=(2, 2))(c4)

  c5 = ReflectionPadding2D(padding=(1,1))(p4)
  c5 = Conv2D(filters=N_ch*16, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c5)
  c5 = BatchNormalization()(c5)  
  c5 = ReflectionPadding2D(padding=(1,1))(c5)
  c5 = Conv2D(filters=N_ch*16, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c5)


  #Expansion path (Decoder)
  u6 = Conv2DTranspose(filters=N_ch*8, kernel_size=(2,2), strides=(2, 2), padding="same")(c5)
  u6 = Concatenate()([u6,c4])
  c6 = ReflectionPadding2D(padding=(1,1))(u6)
  c6 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c6)
  c6 = BatchNormalization()(c6)  
  c6 = ReflectionPadding2D(padding=(1,1))(c6)
  c6 = Conv2D(filters=N_ch*8, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c6)

  u7 = Conv2DTranspose(filters=N_ch*4, kernel_size=(2,2), strides=(2, 2), padding="same")(c6)
  u7 = Concatenate()([u7,c3])
  c7 = ReflectionPadding2D(padding=(1,1))(u7)
  c7 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c7)
  c7 = BatchNormalization()(c7)  
  c7 = ReflectionPadding2D(padding=(1,1))(c7)
  c7 = Conv2D(filters=N_ch*4, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c7)

  u8 = Conv2DTranspose(filters=N_ch*2, kernel_size=(2,2), strides=(2, 2), padding="same")(c7)
  u8 = Concatenate()([u8,c2])
  c8 = ReflectionPadding2D(padding=(1,1))(u8)
  c8 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c8)
  c8 = BatchNormalization()(c8)  
  c8 = ReflectionPadding2D(padding=(1,1))(c8)
  c8 = Conv2D(filters=N_ch*2, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c8)

  u9 = Conv2DTranspose(filters=N_ch, kernel_size=(2,2), strides=(2, 2), padding="same")(c8)
  u9 = Concatenate()([u9,c1])
  c9 = ReflectionPadding2D(padding=(1,1))(u9)
  c9 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c9)
  c9 = BatchNormalization()(c9)  
  c9 = ReflectionPadding2D(padding=(1,1))(c9)
  c9 = Conv2D(filters=N_ch, kernel_size=(3,3), activation='relu', kernel_initializer="he_normal", padding='valid')(c9)

  output = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid',padding="same")(c9)

  model = Model(inputs=input, outputs=output)

  return model
