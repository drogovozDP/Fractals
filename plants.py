import pygame
import numpy as np


FPS = 60
BACK_GROUND = (200, 200, 200)
GRAPH_COLOR = (10, 10, 10)
WIDTH = 900
HEIGHT = 900


class Plot:
    f1 = lambda x: np.array([
        [0.00, 0.00],
        [0.00, 0.16],
    ]) @ x
    f2 = lambda x: np.array([
        [0.85, 0.04],
        [-0.04, 0.85],
    ]) @ x + np.array([0.00, 1.60])
    f3 = lambda x: np.array([
        [0.20, -0.26],
        [0.23, 0.22],
    ]) @ x + np.array([0.00, 1.60])
    f4 = lambda x: np.array([
        [-0.15, 0.28],
        [0.26, 0.24],
    ]) @ x + np.array([0.00, 0.44])

    rules = [f1, f2, f3, f4]
    p = np.array([0.01, 0.85, 0.07, 0.07])

    def __init__(self, n):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.dots = []
        dot = np.array([0.00, 0.00])
        for i in range(n):
            rule = np.random.choice(self.rules, p=self.p)
            dot = rule(dot)
            print(dot)
            self.dots.append(dot)
        self.dots = np.array(self.dots)

    def check_user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def calculate_set(self):
        surface = np.full((HEIGHT, WIDTH, 3), 255)
        WIDTH // 2, HEIGHT // 2
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
    plot = Plot(10000)
    plot.draw()
