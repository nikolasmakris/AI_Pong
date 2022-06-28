import pygame
from random import randint

BLACK = (0, 0, 0)


class Ball(pygame.sprite.Sprite):
    # This class represents a car. It derives from the "Sprite" class in Pygame.

    def __init__(self, color, width, height):
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Pass in the color of the car, and its x and y position, width and height.
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)

        # Draw the ball (a rectangle!)
        pygame.draw.circle(self.image, color, [width/2, height/2], width/2)

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()

        self.BOUNDS = 300

        self.rect.x = self.BOUNDS/2
        self.rect.y = self.BOUNDS/2 - 30
        self.velocity = [randint(-4, 4)*2, randint(-5, -4)]

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]
        self.ball_logic()

    def bounceside(self):
        self.velocity[0] = -self.velocity[0]

    def bounce(self):
        self.velocity[1] = -self.velocity[1]

    def reset(self):
        self.rect.x = self.BOUNDS / 2 - 15
        self.rect.y = self.BOUNDS / 2 - 30
        self.velocity = [randint(-4, 4)*2,  randint(-5, -4)]

    def ball_logic(self):
        if self.rect.x >= self.BOUNDS-30:
            self.rect.x = self.BOUNDS-30
            self.bounceside()
        if self.rect.x <= 0:
            self.rect.x = 0
            self.bounceside()

