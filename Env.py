import cv2
import numpy as np
from PIL import Image
import pygame
from ball import Ball
from paddle import Paddle
from paddleOP import PaddleOP
from collections import deque

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FPS = 60

# open a new window
BOUNDS = 300
win_size = (BOUNDS, BOUNDS)  # size of window
screen = pygame.display.set_mode(win_size)  # Initialize a window or screen for display
pygame.display.set_caption("Pong")

paddleAI = Paddle(WHITE, 50, 10)
paddleOP = PaddleOP(WHITE, 50, 10)

ball = Ball(WHITE, 30, 30)


all_sprites_list = pygame.sprite.Group()

all_sprites_list.add(paddleAI)
all_sprites_list.add(paddleOP)
all_sprites_list.add(ball)


class PongEnv:

    #clock = pygame.time.Clock()

    SIZE = 15
    ACTUAL_SIZE = BOUNDS
    RETURN_IMAGES = True
    LOSE_PENALTY = 20
    WIN_REWARD = 20
    STACK_FRAMES_MEMORY_SIZE = 3
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, STACK_FRAMES_MEMORY_SIZE+1)
    ACTION_SPACE_SIZE = 3



    def __init__(self):
        self.replay_memory = deque(maxlen=self.STACK_FRAMES_MEMORY_SIZE)

    def reset(self):
        ball.reset()
        paddleAI.reset()
        paddleOP.reset()
        self.episode_step = 0  # initializing step
        self.runs = 0  # initializing runs
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        return observation

    def step(self, action):
        self.episode_step += 1
        paddleAI.action(action)
        paddleOP.action(ball.rect.x)
        ball.update()
        if ball.rect.y + ball.velocity[1] < BOUNDS - 30:
            ball.update()
        if pygame.sprite.collide_mask(ball, paddleOP):
            ball.bounce()
            self.runs += 1
        if pygame.sprite.collide_mask(ball, paddleAI):
            ball.bounce()
            self.runs += 1

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())

        if ball.rect.y >= BOUNDS-30:
            reward = -self.LOSE_PENALTY
            ball.reset()
            paddleAI.reset()
            paddleOP.reset()
        elif ball.rect.y < 0:
            reward = self.WIN_REWARD
            ball.reset()
            paddleAI.reset()
            paddleOP.reset()
        else:
            reward = 0

        done = False

        if reward == -self.LOSE_PENALTY or reward == self.WIN_REWARD:
            done = True

        if self.runs >= 10:
            done = True

        return new_observation, reward, done

    def rendersprites(self):
        screen.fill(BLACK)
        all_sprites_list.draw(screen)
        pygame.display.flip()
        #self.clock.tick(FPS)

    def get_image(self):
        env = np.zeros((self.ACTUAL_SIZE, self.ACTUAL_SIZE), dtype=np.uint8)

        for i in range(ball.rect.y, ball.rect.y+30):
            for ii in range(ball.rect.x, ball.rect.x+30):
                if i < 300 and ii < 300:
                    env[(i, ii)] = 255

        for i in range(paddleAI.rect.y, paddleAI.rect.y+10):
            for ii in range(paddleAI.rect.x, paddleAI.rect.x+50):
                if i < 300 and ii < 300:
                    env[(i, ii)] = 255
        #return env

        img = Image.fromarray(env, 'L')  # reading to grayscale.
        img = img.resize((self.SIZE, self.SIZE))
        #img.show()

        current_frame = np.array(img)

        if self.episode_step < 1:
            for i in range(self.STACK_FRAMES_MEMORY_SIZE):
                self.update_stack_image_memory(np.zeros((self.SIZE, self.SIZE), dtype=np.uint8))

        stack_env = np.stack((current_frame, self.replay_memory[2], self.replay_memory[1], self.replay_memory[0]), axis=2)

        self.update_stack_image_memory(current_frame)

        return stack_env

    def update_stack_image_memory(self, frame):
        self.replay_memory.append(frame)






