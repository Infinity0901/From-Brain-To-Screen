from pylsl import StreamInlet, resolve_stream
import time
import tensorflow as tf
import numpy as np
import keyboard


Fs = 256            
n_channels = 4      
Wn = 1            
n_samples = int(Wn * Fs)  

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model_path = 'Virtual Reality EEG/model/model1.h5' 

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
    samples = []


    for _ in range(n_samples):
        sample, _ = inlet.pull_sample()
        samples.append(sample[:-1])

    while True:
        time.sleep(0.05)  
        sample, _ = inlet.pull_sample()
        samples.pop(0) 
        samples.append(sample[:-1])  
 
        x_pred = np.array([samples])
        y_pred = predict(x_pred)

        if y_pred[0][0] > y_pred[0][1] and y_pred[0][0] > y_pred[0][2] and y_pred[0][0] > 0.65:
            print("Gauche")
            keyboard.press('q')
            keyboard.press('z')
            time.sleep(0.2)
            keyboard.release('q')
            keyboard.release('z')
        elif y_pred[0][1] > y_pred[0][0] and y_pred[0][1] > y_pred[0][2] and y_pred[0][1] > 0.65:
            print("Droite")
            keyboard.press('d')
            keyboard.press('z')
            time.sleep(0.2)
            keyboard.release('d')
            keyboard.release('z')
        else:
            print("Neutre")
            keyboard.press('z')
            keyboard.press('z')
            time.sleep(0.2)
            keyboard.release('z')
            keyboard.release('z')
        
        # print(y_pred[0])


if __name__ == '__main__':
    main()

    
    
    
