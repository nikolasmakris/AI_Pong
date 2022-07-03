import pygame
import random

BLACK = (0, 0, 0)


class PaddleOP(pygame.sprite.Sprite):
    # This class represents a paddle. It derives from the "Sprite" class in Pygame.

    def __init__(self, color, width, height):
        # Call the parent class (Sprite) constructor
        super().__init__()

        # Pass in the color of the paddle, its width and height.
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)

        # Draw the paddle (a rectangle!)
        pygame.draw.rect(self.image, color, [0, 0, width, height])

        self.BOUNDS = 300

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()
        self.rect.x = self.BOUNDS/2 - 25
        self.rect.y = 0

        self.difficulty = 6

    def update(self):
        self.move()

    def action(self, ballpos):
        if random.randint(1, self.difficulty) < self.difficulty:
            if ballpos > self.rect.x+15:
                self.move(x=10)
            elif ballpos < self.rect.x-15:
                self.move(x=-10)
            else:
                return
        else:
            self.move(random.randint(-1, 1)*10)

    def move(self, x=False):
        if not x:
            self.rect.x += random.randint(-2, 2)*10
        else:
            self.rect.x += x

        if self.rect.x < 0:
            self.rect.x = 0
        if self.rect.x > self.BOUNDS-50:
            self.rect.x = self.BOUNDS-50


    def reset(self):
        self.rect.x = self.BOUNDS/2 - 25
        self.rect.y = 0
