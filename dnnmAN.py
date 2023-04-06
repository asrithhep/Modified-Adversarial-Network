import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential

class GANregression(keras.Model):
    def __init__(self,ninputs):
        super(GANregression,self).__init__()
        self.discriminator = self.build_discriminator(1)
        self.generator     = self.build_generator(ninputs)

    def build_discriminator(self,ninput):
        dis_input = layers.Input((ninput,))
        label_input = layers.Input((1,))
        merge_input = keras.layers.Concatenate()([dis_input,label_input])

        layer = layers.Dense(256,activation =LeakyReLU(alpha=.5), kernel_initializer='he_uniform')(merge_input)
        layer = layers.Dense(128,activation =LeakyReLU(alpha=.5), kernel_initializer='he_uniform')(layer)
        #layer = layers.Dense(16,activation =LeakyReLU(alpha=.2), kernel_initializer='he_uniform')(layer)
        
        out_layer = layers.Dense(1,activation='sigmoid')(layer)
        model = tf.keras.Model(merge_input,out_layer)
        return model
    def build_generator(self,input_dim):
        model = Sequential()
        model.add(keras.layers.Flatten(input_shape=input_dim))
        hlayer_outline = {'hlayer1':16,'hlayer2':32,'hlayer3':32,'hlayer4':16}
        for layer in hlayer_outline:
            model.add(keras.layers.Dense(hlayer_outline[layer]))
            model.add(LeakyReLU(alpha=0.5))
        model.add(keras.layers.Dense(1,activation='linear',kernel_initializer='normal'))
        return model
    
    def compile(self, d_optimizer, g_optimizer, loss_fn, gen_loss_fn):
        super(GANregression, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.gen_loss_fn   = gen_loss_fn

    def train_step(self,data):
        dnn_train,real_vis_mass= data
        batch_size = tf.shape(dnn_train)[0]
      
        fake_labels = tf.random.normal(shape=(batch_size,1),mean=0.0)       
        generated_vis_mass = self.generator(dnn_train)
      
        fake_vismass_labels = tf.concat([generated_vis_mass,fake_labels],-1)
        real_vismass_labels = tf.concat([real_vis_mass,fake_labels],-1)

        combined_variable = tf.concat([fake_vismass_labels,real_vismass_labels],axis=0)
        labels = tf.concat(
            [tf.ones((batch_size,1)), tf.zeros((batch_size,1))], axis=0
        )
        with tf.GradientTape() as tape:
             predictions = self.discriminator(combined_variable)
             d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
      
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
          generated_vis_mass = self.generator(dnn_train)
          fake_vismass_labels = tf.concat([generated_vis_mass,fake_labels],-1)
          predictions = self.discriminator(fake_vismass_labels)
          g_loss_1 = self.loss_fn(misleading_labels, predictions)
          g_loss_2 = self.gen_loss_fn(real_vis_mass,generated_vis_mass)
          g_loss =g_loss_1 + g_loss_2
        #grads_gen = tape.gradient(gen_loss, self.generator.trainable_weights)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        #grads_tot = grads + grads_gen
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
 

      



class LoadedGANregression(keras.Model):
    def __init__(self,generator,discriminator,ninputs):
        super(LoadedGANregression,self).__init__()
        self.discriminator = discriminator
        self.generator     = generator 
    
    def compile(self, d_optimizer, g_optimizer, loss_fn, gen_loss_fn):
        super(GANregression, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.gen_loss_fn   = gen_loss_fn

    def train_step(self,data):
        dnn_train,real_vis_mass= data
        batch_size = tf.shape(dnn_train)[0]
      
        fake_labels = tf.random.normal(shape=(batch_size,1),mean=0.0)       
        generated_vis_mass = self.generator(dnn_train)
      
        fake_vismass_labels = tf.concat([generated_vis_mass,fake_labels],-1)
        real_vismass_labels = tf.concat([real_vis_mass,fake_labels],-1)

        combined_variable = tf.concat([fake_vismass_labels,real_vismass_labels],axis=0)
        labels = tf.concat(
            [tf.ones((batch_size,1)), tf.zeros((batch_size,1))], axis=0
        )
        with tf.GradientTape() as tape:
             predictions = self.discriminator(combined_variable)
             d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
      
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
          generated_vis_mass = self.generator(dnn_train)
          fake_vismass_labels = tf.concat([generated_vis_mass,fake_labels],-1)
          predictions = self.discriminator(fake_vismass_labels)
          g_loss_1 = self.loss_fn(misleading_labels, predictions)
          g_loss_2 = self.gen_loss_fn(real_vis_mass,generated_vis_mass)
          g_loss =g_loss_1 + g_loss_2
        #grads_gen = tape.gradient(gen_loss, self.generator.trainable_weights)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        #grads_tot = grads + grads_gen
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


