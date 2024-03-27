import pygame
import sys
import csv
from pygame.locals import *
from pylsl import StreamInlet, resolve_stream
import time
import tensorflow as tf
import numpy as np
import threading  


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


model_path = 'Research Interface/model/model1.h5'
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


pygame.init()


pygame.mixer.init()
cloche = pygame.mixer.Sound('Research Interface/sound/bell.ogg')



WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)


SCREEN_WIDTH = 1275
SCREEN_HEIGHT = 730

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("EEG TEST")


font = pygame.font.Font(None, 37)


#===============INTRODUCTION=================


#===============Page 1================= 


text_1_line1 = "Welcome in the  EEG TEST"
text_1 = font.render(text_1_line1 , True, BLACK)
text_1_rect = text_1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 -30))
image1 = pygame.image.load('Research Interface/images/brain_logo.png')
image1 = pygame.transform.scale(image1, (image1.get_width() // 4 -25, image1.get_height() // 4 -25))
image2 = pygame.image.load('Research Interface/images/HL.png')
image2  = pygame.transform.scale(image2, (image2.get_width() // 2, image2.get_height() // 2))

screen.fill(WHITE)
screen.blit(text_1, text_1_rect)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(3000)
# pygame.time.wait(25)




 
#===============Page 2=================


text_2_lines = ["For the test you will place your hands on the table",
                "at 5 cm to an object on your RIGHT and on your LEFT.", 
                "You must remain still ! "]
line_spacing = 15
text_2_rects = []
y_position = SCREEN_HEIGHT // 2 - 250

for line in text_2_lines:
    rendered_line = font.render(line, True, BLACK)
    line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
    text_2_rects.append((rendered_line, line_rect))
    y_position += rendered_line.get_height() + line_spacing  

screen.fill(WHITE)
for rendered_line, line_rect in text_2_rects:
    screen.blit(rendered_line, line_rect)

screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
image3 = pygame.image.load('Research Interface/images/instruction/main.jpg')
image3  = pygame.transform.scale(image3, (image3.get_width() // 2, image3.get_height() // 2))
screen.blit(image3, (SCREEN_WIDTH//2 -500 , SCREEN_HEIGHT//2  -150))
pygame.display.flip()

pygame.time.wait(10000)
# pygame.time.wait(25)


#===============Page 3=================

text_3 = font.render("The test is divided into 3 steps : ", True, BLACK)
text_3_rect = text_3.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

screen.fill(WHITE)
screen.blit(text_3, text_3_rect)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(4000)
# pygame.time.wait(25)



#===============Page 4=================

text_4_lines = ["STEP 1 : ",
                "We will ask you to focus on the object on your ", 
                "RIGHT, LEFT, or in FRONT of you.", 
                "You will have 3 seconds to imagine", 
                "grabbing the object signaled by the two bells.", 
                "And then, the AI will display its prediction."]
line_spacing = 15
text_4_rects = []
y_position = SCREEN_HEIGHT // 2 - 120 

for line in text_4_lines:
    rendered_line = font.render(line, True, BLACK)
    line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
    text_4_rects.append((rendered_line, line_rect))
    y_position += rendered_line.get_height() + line_spacing  

screen.fill(WHITE)
for rendered_line, line_rect in text_4_rects:
    screen.blit(rendered_line, line_rect)
    
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(12000)
# pygame.time.wait(25)

#===============Page 5=================

text_5_lines = ["STEP 2 : ",
                "This time, you will choose the object of focus.",  
                "And then, the AI will display its prediction."]
line_spacing = 15
text_5_rects = []
y_position = SCREEN_HEIGHT // 2 - 120 

for line in text_5_lines:
    rendered_line = font.render(line, True, BLACK)
    line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
    text_5_rects.append((rendered_line, line_rect))
    y_position += rendered_line.get_height() + line_spacing  

screen.fill(WHITE)
for rendered_line, line_rect in text_5_rects:
    screen.blit(rendered_line, line_rect)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(9000)
# pygame.time.wait(25)

#===============Page 6=================

text_6_lines = ["STEP 3 : ",
                "This test will end with a survey"]
line_spacing = 15
text_6_rects = []
y_position = SCREEN_HEIGHT // 2 - 120 

for line in text_6_lines:
    rendered_line = font.render(line, True, BLACK)
    line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
    text_6_rects.append((rendered_line, line_rect))
    y_position += rendered_line.get_height() + line_spacing  

screen.fill(WHITE)
for rendered_line, line_rect in text_6_rects:
    screen.blit(rendered_line, line_rect)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(10000)
# pygame.time.wait(2500)






#===============STEP 1=================

#===============Page 1=================

text_7= font.render("STEP 1   ", True, BLACK)
text_7_rect = text_7.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

text_7_l2 = font.render("Let's start with the first step :  ", True, BLACK)
text_7_rect_l2 = text_7_l2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 -20))



screen.fill(WHITE)
screen.blit(text_7, text_7_rect)
screen.blit(text_7_l2, text_7_rect_l2)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(3000)
# pygame.time.wait(25)

#===============Page 2=================
text_8= font.render("STEP 1   ", True, BLACK)
text_8_rect = text_8.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

text_8_lines = ["As soon as the START message is displayed,", 
                "you will have 3 s to focus on the object."]
line_spacing = 15
text_8_rects = []
y_position = SCREEN_HEIGHT // 2 - 120 

for line in text_8_lines:
    rendered_line = font.render(line, True, BLACK)
    line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
    text_8_rects.append((rendered_line, line_rect))
    y_position += rendered_line.get_height() + line_spacing  

screen.fill(WHITE)
for rendered_line, line_rect in text_8_rects:
    screen.blit(rendered_line, line_rect)
screen.blit(text_8, text_8_rect)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(10000)
# pygame.time.wait(4000)


#===============BOUCLE STEP 1=================

options = ['on your RIGHT', 'on your LEFT', 'in FRONT of you']

num_repetitions = 2

for option in options:
    
    for j in range(num_repetitions):
        #===============Page 3=================

        text_9= font.render("STEP 1  ", True, BLACK)
        text_9_rect = text_9.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

        text_9_lines = ["Focus on the object " + option]
        line_spacing = 15
        text_9_rects = []
        y_position = SCREEN_HEIGHT // 2 - 200

        for line in text_9_lines:
            rendered_line = font.render(line, True, BLACK)
            line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
            text_9_rects.append((rendered_line, line_rect))
            y_position += rendered_line.get_height() + line_spacing

        screen.fill(WHITE)
        for rendered_line, line_rect in text_9_rects:
            screen.blit(rendered_line, line_rect)
        screen.blit(text_9, text_9_rect)
        
        
        droite = pygame.image.load('Research Interface/images/instruction/RIGHT.jpg')
        droite  = pygame.transform.scale(droite, (droite.get_width() // 2, droite.get_height() // 2))
        gauche = pygame.image.load('Research Interface/images/instruction/LEFT.jpg')
        gauche  = pygame.transform.scale(gauche, (gauche.get_width() // 2, gauche.get_height() // 2))
        central = pygame.image.load('Research Interface/images/instruction/CENTRAL.jpg')
        central  = pygame.transform.scale(central, (central.get_width() // 2, central.get_height() // 2))
        # Affichage de l'image en fonction de l'option
        if option == 'on your RIGHT':
            screen.blit(droite,(SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
        elif option == 'on your LEFT':
            screen.blit(gauche, (SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
        elif option == 'in FRONT of you':
            screen.blit(central,(SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
      
        screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
        screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
        pygame.display.flip()
        pygame.time.wait(3000)
        # pygame.time.wait(25)
        
        
        #===============Page 4=================

        text_10= font.render("STEP 1  ", True, BLACK)
        text_10_rect = text_10.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

        text_10_lines = ["START"]
        line_spacing = 15
        text_10_rects = []
        y_position = SCREEN_HEIGHT // 2 - 120

        for line in text_10_lines:
            rendered_line = font.render(line, True, BLACK)
            line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
            text_10_rects.append((rendered_line, line_rect))
            y_position += rendered_line.get_height() + line_spacing

        screen.fill(WHITE)
        for rendered_line, line_rect in text_10_rects:
            screen.blit(rendered_line, line_rect)
        screen.blit(text_10, text_10_rect)
        screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
        screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
        pygame.display.flip()
        cloche.play()
        pygame.time.wait(3000)
        # pygame.time.wait(25)

        #===============Page 5=================

        text_11= font.render("STEP 1  ", True, BLACK)
        text_11_rect = text_11.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

        text_11_lines = ["STOP"]
        line_spacing = 15
        text_11_rects = []
        y_position = SCREEN_HEIGHT // 2 - 120

        for line in text_11_lines:
            rendered_line = font.render(line, True, BLACK)
            line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
            text_11_rects.append((rendered_line, line_rect))
            y_position += rendered_line.get_height() + line_spacing

        screen.fill(WHITE)
        for rendered_line, line_rect in text_11_rects:
            screen.blit(rendered_line, line_rect)
        screen.blit(text_11, text_11_rect)
        screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
        screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
        pygame.display.flip()
        cloche.play()
        pygame.time.wait(1000)
        # pygame.time.wait(25)

        #===============Page 6=================

        text_12= font.render("STEP 1  ", True, BLACK)
        text_12_rect = text_12.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

        text_12_lines = ["The AI predicted " + option]
        line_spacing = 15
        text_12_rects = []
        y_position = SCREEN_HEIGHT // 2 - 200

        for line in text_12_lines:
            rendered_line = font.render(line, True, BLACK)
            line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
            text_12_rects.append((rendered_line, line_rect))
            y_position += rendered_line.get_height() + line_spacing

        screen.fill(WHITE)
        for rendered_line, line_rect in text_12_rects:
            screen.blit(rendered_line, line_rect)
        screen.blit(text_12, text_12_rect)
        
        
        if option == 'on your RIGHT':
            screen.blit(droite,(SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
        elif option == 'on your LEFT':
            screen.blit(gauche, (SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
        elif option == 'in FRONT of you':
            screen.blit(central,(SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
        
        
        screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
        screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
        pygame.display.flip()

        pygame.time.wait(10000)
        # pygame.time.wait(25)
        
        









#===============STEP 2=================


#===============Page 1=================

text_13= font.render("STEP 2   ", True, BLACK)
text_13_rect = text_13.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

text_13_l2 = font.render("Now the second step :  ", True, BLACK)
text_13_rect_l2 = text_13_l2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 -20))



screen.fill(WHITE)
screen.blit(text_13, text_13_rect)
screen.blit(text_13_l2, text_13_rect_l2)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))

pygame.display.flip()

pygame.time.wait(3000)
# pygame.time.wait(25)



#===============Page 2=================


text_14= font.render("STEP 2  ", True, BLACK)
text_14_rect = text_14.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

text_14_lines = ["A message will be displayed asking you to focus on the object",
                "of YOUR choice.",
                "Then, the AI will displayed its prediction."]
line_spacing = 15
text_14_rects = []
y_position = SCREEN_HEIGHT // 2 - 120 

for line in text_14_lines:
    rendered_line = font.render(line, True, BLACK)
    line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
    text_14_rects.append((rendered_line, line_rect))
    y_position += rendered_line.get_height() + line_spacing  

screen.fill(WHITE)
for rendered_line, line_rect in text_14_rects:
    screen.blit(rendered_line, line_rect)
    
    
screen.blit(text_14, text_14_rect)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(10000)
# pygame.time.wait(25)


#===============STEP 2=================

for i in range(6): 
    
    
    #===============Page 3=================

    text_15= font.render("STEP 2  ", True, BLACK)
    text_15_rect = text_15.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

    text_15_lines = ["Focus on the object on your choice"]
    line_spacing = 15
    text_15_rects = []
    y_position = SCREEN_HEIGHT // 2 - 200 

    for line in text_15_lines:
        rendered_line = font.render(line, True, BLACK)
        line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
        text_15_rects.append((rendered_line, line_rect))
        y_position += rendered_line.get_height() + line_spacing  

    screen.fill(WHITE)
    for rendered_line, line_rect in text_15_rects:
        screen.blit(rendered_line, line_rect)
        
        
    screen.blit(text_15, text_15_rect)
    screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
    screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
    screen.blit(image3, (SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
    
    pygame.display.flip()

    pygame.time.wait(5000)
    # pygame.time.wait(25)


    #===============Page 4=================


    text_16= font.render("STEP 2  ", True, BLACK)
    text_16_rect = text_16.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

    text_16_lines = ["START"]
    line_spacing = 15
    text_16_rects = []
    y_position = SCREEN_HEIGHT // 2 - 120 

    for line in text_16_lines:
        rendered_line = font.render(line, True, BLACK)
        line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
        text_16_rects.append((rendered_line, line_rect))
        y_position += rendered_line.get_height() + line_spacing  

    screen.fill(WHITE)
    for rendered_line, line_rect in text_16_rects:
        screen.blit(rendered_line, line_rect)
        
        
    screen.blit(text_16, text_16_rect)
    screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
    screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
    pygame.display.flip()
    cloche.play()
    pygame.time.wait(3000)
    # pygame.time.wait(25)


    #===============Page 5=================

    text_17= font.render("STEP 2  ", True, BLACK)
    text_17_rect = text_17.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

    text_17_lines = ["STOP"]
    line_spacing = 15
    text_17_rects = []
    y_position = SCREEN_HEIGHT // 2 - 120 

    for line in text_17_lines:
        rendered_line = font.render(line, True, BLACK)
        line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
        text_17_rects.append((rendered_line, line_rect))
        y_position += rendered_line.get_height() + line_spacing  

    screen.fill(WHITE)
    for rendered_line, line_rect in text_17_rects:
        screen.blit(rendered_line, line_rect)
        
        
    screen.blit(text_17, text_17_rect)
    screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
    screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
    pygame.display.flip()
    cloche.play()
    pygame.time.wait(2000)
    # pygame.time.wait(25)



    #===============Page 6=================

    text_18= font.render("STEP 2  ", True, BLACK)
    text_18_rect = text_18.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

    text_18_lines = ["The AI predicted : " + prediction_text]
    line_spacing = 15
    text_18_rects = []
    y_position = SCREEN_HEIGHT // 2 - 200

    for line in text_18_lines:
        rendered_line = font.render(line, True, BLACK)
        line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
        text_18_rects.append((rendered_line, line_rect))
        y_position += rendered_line.get_height() + line_spacing  

    screen.fill(WHITE)
    for rendered_line, line_rect in text_18_rects:
        screen.blit(rendered_line, line_rect)
        
    if prediction_text == "RIGHT":
            screen.blit(droite,(SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
    elif prediction_text == "LEFT":
            screen.blit(gauche, (SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
    elif prediction_text == "NEUTRAL":
            screen.blit(central,(SCREEN_WIDTH//2 -475 , SCREEN_HEIGHT//2  -150))
        
        
        
    screen.blit(text_18, text_18_rect)
    screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
    screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
    pygame.display.flip()

    pygame.time.wait(10000)
    # pygame.time.wait(25)



#===============QUESTIONNARY=================

#===============Page 1=================

text_19= font.render("STEP 3   ", True, BLACK)
text_19_rect = text_19.get_rect(center=(SCREEN_WIDTH // 2 , SCREEN_HEIGHT // 2 -320))

text_19_lines = ["SURVEY : ", 
                 "Please answer and submit !"]
line_spacing = 15
text_19_rects = []
y_position = SCREEN_HEIGHT // 2 - 120 

for line in text_19_lines:
    rendered_line = font.render(line, True, BLACK)
    line_rect = rendered_line.get_rect(center=(SCREEN_WIDTH // 2, y_position))
    text_19_rects.append((rendered_line, line_rect))
    y_position += rendered_line.get_height() + line_spacing  

screen.fill(WHITE)
for rendered_line, line_rect in text_19_rects:
    screen.blit(rendered_line, line_rect)

screen.blit(text_19, text_19_rect)
screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
pygame.display.flip()

pygame.time.wait(7500)
# pygame.time.wait(25)



# ===============Page 2=================

questions = [
    {"question": "How many mistakes made the AI Model ? ", "options": ["a) Between 1 and 3", "b) Between 4 and 6 ", "c) More"]},
    {"question": "Were you focus all along the test ? ", "options": ["a) Yes", "b) No", "c) I think so"]},
    {"question": "Did you understand the dynamics of the test quickly ? ", "options": ["a) Yes  ", "b) No", "c)I think so "]},
    {"question": "What did you think of this test ? ", "options": ["a) Perfect ", "b) Good enough", "c) Could be better"]}
]


class CheckBox:
    def __init__(self, x, y, text):
        self.x = x
        self.y = y
        self.text = text
        self.checked = False

    def draw(self, screen):
        checkbox_size = 20  
        checkbox_rect = pygame.Rect(self.x, self.y, checkbox_size, checkbox_size)
        
        pygame.draw.rect(screen, BLACK, checkbox_rect, 2)
        
        if self.checked:
            pygame.draw.rect(screen, BLACK, checkbox_rect.inflate(-10, -10))
        
        screen.blit(font.render(self.text, True, BLACK), (self.x + checkbox_size + 10, self.y))

    def toggle(self):
        self.checked = not self.checked

checkboxes = []
for i, question in enumerate(questions):
    for j, option in enumerate(question["options"]):
        checkboxes.append(CheckBox(100, 100 + (i * 150) + (j * 50), option))

submit_button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 100, 200, 50)
submit_button_text = font.render("Submit", True, BLACK)
pygame.display.flip()





# ===============MAIN=================

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for checkbox in checkboxes:
                checkbox_rect = pygame.Rect(checkbox.x, checkbox.y, 20, 20)
                if checkbox_rect.collidepoint(mouse_pos):
                    checkbox.toggle()
            if submit_button_rect.collidepoint(mouse_pos):
                with open('reponses.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Questions', 'Reponses'])
                    for i, checkbox in enumerate(checkboxes):
                        question = questions[i // 3]["question"]
                        response = questions[i // 3]["options"][i % 3]
                        if checkbox.checked:
                            writer.writerow([question, response])
                running = False

    screen.fill(WHITE)
    
    max_question_width = max([font.size(question["question"])[0] for question in questions])

    for i, question in enumerate(questions):
        # Dsiplay the centered question
        question_text = font.render(question["question"], True, BLACK)
        question_rect = question_text.get_rect()
        question_rect.centerx = SCREEN_WIDTH // 2
        question_rect.top = 50 + i * 150
        screen.blit(question_text, question_rect)

        # Display the boxes
        for j, option in enumerate(question["options"]):
            checkbox = checkboxes[i * 3 + j]
            checkbox.draw(screen)
            checkbox.x = question_rect.right + 10
            checkbox.y = question_rect.top + j * 50
            
    # Display the submit button
    pygame.draw.rect(screen, GRAY, submit_button_rect)
    screen.blit(image1,  (SCREEN_WIDTH// 4 +850,  SCREEN_HEIGHT // 4 -180))
    screen.blit(image2,  (SCREEN_WIDTH// 4 -300,  SCREEN_HEIGHT // 4 -180))
    screen.blit(submit_button_text, submit_button_rect.move(50, 10))
    
    pygame.display.flip()

pygame.quit()
sys.exit()




                

