import pygame
import numpy as np


FPS = 60
BACK_GROUND = (200, 200, 200)
GRAPH_COLOR = (10, 10, 10)
WIDTH = 900
HEIGHT = 900


class Plot:
    def __init__(self, fun):
        pygame.init()
        self.fun = fun
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.c = 0.5481 + 0.5481j
        self.surface = np.zeros((WIDTH, HEIGHT, 3), dtype=complex)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                self.surface[y, x] = x + y * (-1) ** (1 / 2)
        self.surface -= ((WIDTH // 2) + (HEIGHT // 2) * (-1) ** (1 / 2))
        self.min_val = 100 + 100j
        self.koef = 3e-3

    def check_user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def calculate_set(self):
        surface = np.full((HEIGHT, WIDTH, 3), 255)
        n = 200
        """
        SIMPLEST SOLUTION
        In this solution we have approximately 0.35 FPS with n=100
        """
        # half_y, half_x = HEIGHT // 2, WIDTH // 2
        # for y_i in range(HEIGHT):
        #     for x_i in range(WIDTH):
        #         x = (x_i - half_x) * koef
        #         y = (y_i - half_y) * koef
        #         z = x + y * (-1)**(1/2)
        #         for i in range(n):
        #             if abs(z) > 1000:
        #                 break
        #             z = self.fun(z, self.c)
        #         z = abs(z)
        #         z = 1 if z > 1 else z
        #         surface[y_i, x_i] = surface[y_i, x_i] * z
        """
        MATRIX SOLUTION
        In this solution we have approximately 2.6 FPS with n=100
        """
        scaled_surface = self.surface * self.koef
        for i in range(n):
            scaled_surface = self.fun(scaled_surface, self.c)

        scaled_surface = abs(scaled_surface)
        scaled_surface[scaled_surface > 1] = 1
        scaled_surface[scaled_surface <= 1] = 1
        surface = surface * scaled_surface
        # self.koef *= 0.9
        return surface

    def update_screen(self):
        self.screen.fill(BACK_GROUND)
        surface = pygame.surfarray.make_surface(self.calculate_set())
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def draw(self):
        while self.running:
            self.clock.tick(FPS)
            self.check_user_input()
            self.update_screen()
            pygame.display.set_caption(f"FPS={self.clock.get_fps()}")


if __name__ == "__main__":
    fun = lambda z, c: z**(4) + c
    plot = Plot(fun)
    plot.draw()
