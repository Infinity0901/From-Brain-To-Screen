from pylsl import StreamInlet, resolve_stream
import time
import tensorflow as tf
import numpy as np
import pandas as pd

Fs = 256            
n_channels = 4      
Wn = 1             
n_samples = Wn * Fs  

def load_model(model_path):
    return tf.keras.models.load_model(model_path)


model_path = 'main/model/model1.h5' 

test_model = load_model(model_path)

def predict(x_pred):
    x_pred = np.transpose(x_pred, (0, 2, 1))
    x_pred = x_pred[:, :, :, np.newaxis]
    y_pred = test_model.predict(x_pred)
    return y_pred



def main():
   
    print("Recherche d'un flux EEG...")
    streams = resolve_stream('type', 'EEG')

    inlet = StreamInlet(streams[0])

    while True:
        start_time = time.time() 
        samples = []
        for _ in range(n_samples):
            sample, _ = inlet.pull_sample()
            samples.append(sample[:-1])  
        
          

        #  Prediction toutes les secondes
        if time.time() - start_time >=1:
            x_pred = np.array([samples])
            y_pred = predict(x_pred)

            if y_pred[0][0] > y_pred[0][1] and y_pred[0][0] > y_pred[0][2] and y_pred[0][0] > 0.85:
                print('Prédiction : Gauche'.format(y_pred[0][0]))
                
                
            elif y_pred[0][1] > y_pred[0][0] and y_pred[0][1] > y_pred[0][2] and y_pred[0][1] > 0.95:
                print('Prédiction : Droite'.format(y_pred[0][1]))
                 
            else:
                print('Prédiction : Neutre'.format(y_pred[0][2]))
                
            

if __name__ == '__main__':
    main()
    