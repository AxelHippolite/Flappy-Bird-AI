import pygame
from pygame.locals import * 
import sys
import numpy as np 
import random
import math

np.seterr(all='ignore')

class FlappyBird:
    def __init__(self):   
        self.screen = pygame.display.set_mode((400, 700))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("Assets/background.png").convert()
        self.birdSprites = [pygame.image.load("Assets/1.png").convert_alpha(),
                            pygame.image.load("Assets/2.png").convert_alpha(),
                            pygame.image.load("Assets/dead.png")]
        self.wallUp = pygame.image.load("Assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("Assets/top.png").convert_alpha()
        self.gap = 130
        self.wallx = 400
        self.birdY = 200
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.deadVerif = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-110, 110)

    def initialization(self):
        THETA1_rows = 5
        THETA1_col = 3
        THETA2_rows = 1
        THETA2_col = 6
        self.N = 10
        self.distance = 0
        self.generation = 0
        self.pred = 0
        self.pc = 0.8
        self.pm = 0.3
        self.nbGene = 21
        self.fitness_score = np.zeros((self.N,1))
        self.hidden_layer = 5
        self.thresh = 10
        self.unrollRows = (THETA1_rows * THETA1_col) + (THETA2_rows * THETA2_col)
        self.population = np.zeros((self.N, (THETA1_rows * THETA1_col + THETA2_rows * THETA2_col)))
        self.THETA1 = np.zeros((THETA1_rows, THETA1_col))
        self.THETA2 = np.zeros((THETA2_rows, THETA2_col))

        for u in range(self.N):
            for i in range(THETA1_rows):
                for t in range(THETA1_col):
                    self.THETA1[i,t] = random.uniform(-self.thresh, self.thresh)
                    t = t + 1
                i = i + 1

            for i in range(THETA2_rows):
                for t in range(THETA2_col):
                    self.THETA2[i,t] = random.uniform(-self.thresh, self.thresh)
                    t = t + 1
                i = i + 1

            self.THETA1 = self.THETA1.reshape(15, 1)
            self.THETA2 = self.THETA2.T
            genes = np.concatenate((self.THETA1, self.THETA2), axis = 0)
            genes = genes.T
            self.population[u,:] = genes
            self.THETA1 = np.zeros((THETA1_rows, THETA1_col))
            self.THETA2 = np.zeros((THETA2_rows, THETA2_col))
            u = u + 1

    def mutation(self): #Gene Mutation Part
        for i in range(self.N):
            jesusHand = random.uniform(1, 100)
            chro = random.randint(1, self.N - 1)
            g = random.randint(1, self.nbGene - 1)
            if jesusHand <= (self.pm * 100):
                newGene = random.uniform(-self.thresh, self.thresh)
                self.population[chro, g] = newGene
            else:
                self.population = self.population
            i = i + 1
        return self.population
    
    def crossover(self, parent1, parent2): #Crossover Part
        jesusHand = random.uniform(0, 100)
        if jesusHand <= (self.pc * 100):
            crossPoint = random.randint(1, self.nbGene)
            genTransParent1 = parent1[crossPoint:self.nbGene]#elite
            genTransParent2 = parent2[crossPoint:self.nbGene]#elite
            fille1 = np.zeros((1,self.nbGene))
            fille2 = np.zeros((1,self.nbGene))
            for i in range(crossPoint):
                fille1[:, i] = parent1[i]
                i = i + 1
            for i in range(self.nbGene - crossPoint):
                a = i + crossPoint
                fille1[:, a] = genTransParent2[i]
                i = i + 1
            for i in range(crossPoint):
                fille2[:, i] = parent2[i]
                i = i + 1
            for i in range(self.nbGene - crossPoint):
                a = i + crossPoint
                fille2[:, a] = genTransParent1[i]
                i = i + 1
            pos_dead1, pos_dead2 = self.deadselection()
            self.population[pos_dead1, :] = fille1
            self.population[pos_dead2, :] = fille2
        else:
            self.popuation = self.population
        return self.population
    
    def selection(self): #Parents Selection
        max1_fit = max(self.fitness_score)
        max1_fit_replace = float(max1_fit)
        max1_fit_pos = np.where(self.fitness_score == max1_fit)
        row1 = int(max1_fit_pos[0][0])
        parent1 = self.population[row1, :]
        self.fitness_score[row1, :] = 0
        max2_fit = max(self.fitness_score)
        max2_fit_pos = np.where(self.fitness_score == max2_fit)
        row2 = int(max2_fit_pos[0][0])
        parent2 = self.population[row2, :]
        self.fitness_score[row1, :] = max1_fit_replace
        return parent1, parent2

    def deadselection(self): #Dead Members Selection
        min_fit = min(self.fitness_score)
        min_fit_pos = np.where(self.fitness_score == min_fit)
        pos_dead1 = int(min_fit_pos[0][0])
        self.fitness_score[pos_dead1, :] = 0
        min_fit2 = min(self.fitness_score)
        min_fit2_pos = np.where(self.fitness_score == min_fit2)
        pos_dead2 = int(min_fit2_pos[0][0])
        return pos_dead1, pos_dead2
    
    def fitnessAssigment(self, indiv, input_h): #Fitness Score Assigment
        self.fitness_score[indiv,:] = self.counter + self.distance - self.ptsDistance - input_h
        self.distance = 0
        return self.fitness_score
    
    def sigmoid(self, Z):
        g = 1.0 / (1.0 + np.exp(-Z))
        return g

    def nn_flap(self, individual, P): #NeuralNet Prediction
        THETA1_col = 3
        individual = individual.T
        Theta1 = individual[0:15]
        Theta2 = individual[15:21]
        Theta1 = Theta1.reshape(5,3)
        Theta2 = Theta2.T
        bias = np.zeros((1,THETA1_col))
        bias[0,0] = 1
        for i in range(THETA1_col - 1):
            a = i + 1
            bias[:, a] = P[:,i]
            i = i + 1
        Z = np.dot(bias, Theta1.T)
        h1 = self.sigmoid(Z)
        bias = np.zeros((1,(self.hidden_layer + 1)))
        bias[0,0] = 1
        for i in range(self.hidden_layer):
            a = i + 1
            bias[:, a] = h1[:, i]
            i = i + 1
        Z = np.dot(bias, Theta2.T)
        h2 = self.sigmoid(Z)
        if h2 > 0.5:
            self.pred = 1
        else:
            self.pred = 0
        return self.pred
    
    def updateWalls(self):
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-110, 110)

    def birdUpdate(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        if upRect.colliderect(self.bird):
            self.deadVerif = True
            self.bird[1] = 50
            self.birdY = 200
            self.dead = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5
        if downRect.colliderect(self.bird):
            self.deadVerif = True
            self.bird[1] = 50
            self.birdY = 200
            self.dead = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5
        if not 0 < self.bird[1] < 720:
            self.bird[1] = 50
            self.birdY = 200
            self.dead = False
            self.deadVerif = True
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5
        #if self.counter >= 500:
        #    self.bird[1] = 50
        #    self.birdY = 200
        #    self.dead = False
        #    self.deadVerif = True
        #    self.counter = 0
        #    self.wallx = 400
        #    self.offset = random.randint(-110, 110)
        #    self.gravity = 5

    def run(self):
        self.initialization()
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        while True:
            for indiv in range(self.N):
                individual = self.population[indiv,:]
                while self.deadVerif != True:
                    clock.tick(60)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                        if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                            self.jump = 17
                            self.gravity = 5
                            self.jumpSpeed = 10
                    if self.pred == 1 and not self.dead:
                        self.jump = 17
                        self.gravity = 5
                        self.jumpSpeed = 10

                    self.screen.fill((255, 255, 255))
                    self.screen.blit(self.background, (0, 0))
                    self.screen.blit(self.wallUp,
                                     (self.wallx, 360 + self.gap - self.offset))
                    self.screen.blit(self.wallDown,
                                     (self.wallx, 0 - self.gap - self.offset))
                    self.screen.blit(font.render(str(self.counter),
                                                 -1,
                                                 (255, 255, 255)),
                                     (200, 50))
                    self.screen.blit(font.render(str(indiv + 1),
                                                 -1,
                                                 (255, 255, 255)),
                                     (75, self.birdY - 60))
                    if self.dead:
                        self.sprite = 2
                    elif self.jump:
                        self.sprite = 1
                    self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
                    if not self.dead:
                        self.sprite = 0
                    self.updateWalls()
                    self.birdUpdate()
                    self.distance = self.distance + 5
                    x_pts = self.wallx + 95
                    y_pts = 360 + self.gap - self.offset - 60
                    x_bird = 105
                    y_bird = int(self.birdY)
                    input_d = x_pts - x_bird
                    input_h = y_pts - y_bird
                    P = np.zeros((1,2))
                    P[:, 0] = input_d
                    P[:, 1] = input_h
                    self.ptsDistance = (input_d**2 + input_h**2)**0.5
                    self.pred = self.nn_flap(individual, P)
                    #print("Individu : " + str(indiv))
                    pygame.display.update()
                self.fitness_score = self.fitnessAssigment(indiv, input_h)
                fitness_mean = np.mean(self.fitness_score)
                indiv = indiv + 1
                self.distance = 0
                self.deadVerif = False
            self.generation = self.generation + 1
            print("##################")
            print("Generation : " + str(self.generation))
            parent1, parent2 = self.selection()
            print("Fitness Score Mean : " + str(fitness_mean))
            print("**Selection***")
            population = self.crossover(parent1, parent2)
            print("**Crossover***")
            population = self.mutation()
            print("**Mutation***")
            print("##################\n")
            self.fitness_score = np.zeros((self.N,1))
            indiv = 0

if __name__ == "__main__":
    FlappyBird().run()
