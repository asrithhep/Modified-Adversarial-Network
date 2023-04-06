import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential

from sklearn.utils import shuffle






from dnnmAN import GANregression
from dnnmAN import LoadedGANregression




num_pipeline = Pipeline([('MinMaxScaler',MinMaxScaler(feature_range=(0,1))) ])

num_pipeline1 = Pipeline([('MinMaxScaler',MinMaxScaler(feature_range=(0,1))) ])

    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-tr","--train",help="to train the model",action="store_true")
parser.add_argument("-ts","--test",help="to test the model",action="store_true")
parser.add_argument("-b","--back",help="background boolean")
args = parser.parse_args()
    




class GANMonitor(keras.callbacks.Callback):
    def __init__(self,real_tau_p4,vis_mass,prefix,incriment):
        self.real_tau_p4 = real_tau_p4
        self.incriment = incriment
        self.prefix = prefix
        self.vis_mass = vis_mass



    def summarise_performance(self,real_boson_mass,fake_boson_mass):
        fig,ax = plt.subplots()
        (n1, bins1, patches1)=ax.hist(real_boson_mass,20,color='blue',label='real')
        (n2, bins2, patches2)=ax.hist(fake_boson_mass,20,color='red',alpha=0.5,label='fake')
       
        ax.legend()

    def SaveModels(self,epoch):
        discriminator = self.model.discriminator
        generator = self.model.generator
        discriminator.trainable = False
        dir_path = './train/MAN_256-128/3tev/'
        tf.keras.models.save_model(generator,dir_path+'generator_'+self.prefix+'_'+str(epoch))
        tf.keras.models.save_model(discriminator,dir_path+'discriminator_'+self.prefix+'_'+str(epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        generated_vismass_labels = self.real_tau_p4
        self.fake_vismass  = self.model.generator(generated_vismass_labels)
        if epoch % self.incriment == 0 and epoch != 0:
          self.summarise_performance(self.vis_mass,self.fake_vismass[:,0])
          self.SaveModels(epoch)  


TrainDataSet  = pd.read_csv('./data/train_3tev.csv',index_col=None ) 
TrainDataSet.fillna(value=TrainDataSet.mean(),inplace=True)
#TrainDataSet = TrainDataSet[:5000]


TestDataSet = pd.read_csv('./data/test3tev_sig.csv',index_col=False) 
TestDataSet.fillna(value=TestDataSet.mean(),inplace=True)
#TestDataSet = TestDataSet[:1000]

if(args.back == True):
   TestDataSet = pd.read_csv('./data/test_back.csv',index_col=False)
   TestDataSet.fillna(value=TestDataSet.mean(),inplace=True)
   #TestDataSet = TestDataSet[:1000]



training_features = TrainDataSet.drop(columns=['gentau_vis_pt','gentau_vis_eta','gentau_vis_phi','neutrino_px','neutrino_py','neutrino_pz','neutrino_e','nu_px','nu_py','nu_pz','nu_e','genmet','genmet_phi','boson_mass']).to_numpy(dtype='float32')
training_labels  = TrainDataSet['boson_mass'].to_numpy(dtype='float32')

xstar  = TestDataSet.drop(columns=['gentau_vis_pt','gentau_vis_eta','gentau_vis_phi','neutrino_px','neutrino_py','neutrino_pz','neutrino_e','nu_px','nu_py','nu_pz','nu_e','genmet','genmet_phi','boson_mass']).to_numpy(dtype='float32')
ystar  = TestDataSet['boson_mass'].to_numpy(dtype='float32')   



xtrain_star = np.concatenate((training_features,xstar),axis=0)
ytrain_star = np.concatenate((training_labels,ystar),axis=0)


xtrain_star_tr = num_pipeline.fit_transform(xtrain_star)  
ytrain_star_tr = num_pipeline.fit_transform(ytrain_star.reshape(-1,1))



xtrain__tr = num_pipeline1.fit_transform(training_features)  
ytrain__tr = num_pipeline1.fit_transform(training_labels.reshape(-1,1))


xtrain_tr = xtrain_star_tr[:len(training_features)]
xstar_tr = xtrain_star_tr[len(training_features):]

ytrain_tr = ytrain_star_tr[:len(training_features)]
ystar_tr = ytrain_star_tr[len(training_features):]



gan = GANregression(ninputs=(6,))
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=False),
    gen_loss_fn = keras.losses.MeanSquaredError()
   )
if(args.train == True):
   gan.fit(xtrain_tr,ytrain_tr,callbacks=[GANMonitor(xtrain_tr,ytrain_tr,'Wprime',1000)],epochs=10001,verbose=True)
   _ganW_mass = gan.generator(xstar_tr)
   y_pred = num_pipeline.inverse_transform(_ganW_mass)
   df = pd.DataFrame({"regression_mass" : y_pred[:,0], "boson_mass" :ystar})
   if(args.back == True):
      df.to_csv('background3.csv', index=False)
   else:
      df.to_csv('signal3.csv', index=False)

if(args.test == True):
   discriminator = tf.keras.models.load_model('./train/MAN_256-128/3tev/discriminator_Wprime_8000')
   generator     = tf.keras.models.load_model('./train/MAN_256-128/3tev/generator_Wprime_8000')
   load_gan = LoadedGANregression(generator=generator,discriminator=discriminator,ninputs=(6,))
   _ganW_mass = load_gan.generator(xstar_tr)
   y_pred = num_pipeline.inverse_transform(_ganW_mass)
   df = pd.DataFrame({"regression_mass" : y_pred[:,0], "boson_mass" :ystar})
   if(args.back == True):
      df.to_csv('background3.csv', index=False)
   else:
      df.to_csv('signal3.csv', index=False)



