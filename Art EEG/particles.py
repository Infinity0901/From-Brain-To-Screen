import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
from pygame.locals import *
from pygame import mixer
import tensorflow as tf
import threading  
import numpy as np
from pylsl import StreamInlet, resolve_stream
import time

Fs = 256            
n_channels = 4      
Wn = 1             
n_samples = Wn * Fs  

def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict(x_pred):
    x_pred = np.transpose(x_pred, (0, 2, 1))
    x_pred = x_pred[:, :, :, np.newaxis]
    y_pred = test_model.predict(x_pred)
    return y_pred


model_path = 'Art EEG/model/model1.h5'
test_model = load_model(model_path)


prediction_text = ""
def predict_data():
    global prediction_text
    print("Recherche d'un flux EEG...")
    streams = resolve_stream('type', 'EEG')

    inlet = StreamInlet(streams[0])
    while True:
        start_time = time.time() 
        samples = []
        for _ in range(n_samples):
            sample, _ = inlet.pull_sample()
            samples.append(sample[:-1])  
        if time.time() - start_time >= 1:
            x_pred = np.array([samples])
            y_pred = predict(x_pred)
            if y_pred[0][0] > y_pred[0][1] and y_pred[0][0] > y_pred[0][2] and y_pred[0][0] > 0.85:
                prediction_text = "LEFT"
            elif y_pred[0][1] > y_pred[0][0] and y_pred[0][1] > y_pred[0][2] and y_pred[0][1] > 0.95:
                prediction_text = "RIGHT"
            else:
                prediction_text = "NEUTRAL"


prediction_thread = threading.Thread(target=predict_data)
prediction_thread.daemon = True
prediction_thread.start()


class Particle:
    def __init__(self):
        self.position = [0, 0, 0]
        self.velocity = [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]

    def update(self):
        for i in range(3):
            self.position[i] += self.velocity[i]

    def draw(self):
        glBegin(GL_POINTS)
        glColor3f(1, 1, 1)
        glVertex3fv(self.position)
        glEnd()


pygame.init()

WIDTH, HEIGHT = 1920 , 1300

screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)



gluPerspective(45, (WIDTH/HEIGHT), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

particles = []

PARTICLES_PER_KEY_PRESS = 25

# Boucle principale
clock = pygame.time.Clock()

pygame.mixer.init()
pygame.mixer.music.load('music/music.mp3')
pygame.mixer.music.play(-1)  

while True:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    if prediction_text == "NEUTRAL":
        pass
    elif prediction_text == "LEFT":
        glTranslatef(-0.001, 0, 0)
        for _ in range(PARTICLES_PER_KEY_PRESS):
            particles.append(Particle())
    elif prediction_text == "RIGHT":
        glTranslatef(0.001, 0, 0)
        for _ in range(PARTICLES_PER_KEY_PRESS):
            particles.append(Particle())


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    for particle in particles:
        particle.update()
        particle.draw()
    
    pygame.display.flip()
    clock.tick(60)
