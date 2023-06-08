import tensorflow as tf
keras=tf.keras


valid_model_types = ["mlp", "conv2d", "resnet"]

class ModelV2:
  """The model."""

  def __init__(self,input_shape, max_size,max_actions, config,architecture="mlp"):
    # game params
    self.action_size = max_actions
    self.config = config

    # Neural Net
    match architecture:
        case "mlp":
            self.input_state = keras.layers.Input(shape=input_shape,name="environment")    # s: batch_size x board_x x board_y
            flattened=keras.layers.Flatten(name="flatten")(self.input_state)
            y=keras.layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(1),gamma_regularizer=keras.regularizers.l2(1))(flattened)
            y=keras.layers.Dense(128,activation="relu",kernel_regularizer=keras.regularizers.l2(1))(y)
            z=keras.layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(1),gamma_regularizer=keras.regularizers.l2(1))(y)
            self.pi=keras.layers.Dense(self.action_size,activation="softmax",name="pi",kernel_regularizer=keras.regularizers.l2(1))(z)   # batch_size x self.action_size
            self.v=keras.layers.Dense(1,activation="tanh",name="v",kernel_regularizer=keras.regularizers.l2(1))(z)                    # batch_size x 1
            self.model=keras.Model(inputs=self.input_state,outputs=[self.pi,self.v])
        case "mpgnet":
            self.input_environment = keras.layers.Input(shape=input_shape,name="environment")    # s: batch_size x board_x x board_y
            self.input_state=keras.layers.Input(shape=(),name="state")
            state_reshape=keras.layers.Reshape((1,))(self.input_state)
            flattened=keras.layers.Flatten(name="flatten")(self.input_environment)
            stack=keras.layers.Concatenate()([flattened,state_reshape])
            y=keras.layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(1),gamma_regularizer=keras.regularizers.l2(1))(stack)
            y=keras.layers.Dense(128,activation="relu",kernel_regularizer=keras.regularizers.l2(1))(y)
            z=keras.layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(1),gamma_regularizer=keras.regularizers.l2(1))(y)
            self.pi=keras.layers.Dense(self.action_size,activation="softmax",name="pi",kernel_regularizer=keras.regularizers.l2(1))(z)   # batch_size x self.action_size
            self.v=keras.layers.Dense(1,activation="tanh",name="v",kernel_regularizer=keras.regularizers.l2(1))(z)                    # batch_size x 1
            self.model=keras.models.Model(inputs=[self.input_environment,self.input_state],outputs=[self.pi,self.v])
            self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=keras.optimizers.Adam(config.lr))

  @classmethod
  def from_checkpoint(cls,path:str):
      """
      Builds a graph from a checkpoint
      :param path: path to the checkpoint
      :return: ModelV2
      """
      instance=ModelV2.__new__(cls)
      instance.model=keras.models.load_model(path)
      instance.input_environment=instance.model.get_layer("environment").input
      instance.input_state=instance.model.get_layer("state").input
      instance.pi=instance.model.get_layer("pi").output
      instance.v=instance.model.get_layer("v").output
      return instance

  @classmethod
  def from_config(cls,config):
    raise NotImplementedError("from_config not implemented")

  def save_checkpoint(self,path:str):
    """
    Saves the model to a checkpoint
    :param path: path to the checkpoint
    """
    self.model.save(path)

  def predict(self,env,state):
    """
    :param env: the environment
    :param state: the state
    :return: pi, v
    """
    return self.model.predict([env,state])

  def loss_function(self):
      return self.model.loss

