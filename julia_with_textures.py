import pygame
import numpy as np
from numba import njit


FPS = 60
BACK_GROUND = (200, 200, 200)
GRAPH_COLOR = (10, 10, 10)
WIDTH = 900
HEIGHT = 900
DX = 10
CX = 1e-1

texture = pygame.image.load("img/texture_3.jpg")
TEXTURE_SIZE = min(texture.get_size()) - 1
TEXTURE_ARRAY = pygame.surfarray.array3d(texture)


class Plot:
    keys = [
        pygame.K_UP,
        pygame.K_DOWN,
        pygame.K_LEFT,
        pygame.K_RIGHT,
        pygame.K_p,
        pygame.K_m,
        pygame.K_z,
        pygame.K_x,
        pygame.K_a,
        pygame.K_s,
    ]
    key_pressed = {key: False for key in keys}

    def __init__(self, fun):
        pygame.init()
        self.fun = fun
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.surface = np.full((HEIGHT, WIDTH, 3), 255)
        self.plain_surface = np.zeros((HEIGHT, WIDTH, 3))
        self.offset = np.array([0, 0], dtype=float)
        self.center = np.array([0, 0])
        self.koef = 3e-3
        self.dx = DX * self.koef
        # self.c = 0.35 + 0.35j
        # self.c = 0.0 - 0.65j
        self.c = 0.4 + 0.4j

    @staticmethod
    @njit(fastmath=True)
    def calculate_fractal(screen_array, X, Y, c, max_iter):
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                z = x + 1j * y
                num_iter = 0
                for _ in range(max_iter):
                    z = z*z + c
                    if z.real**2 + z.imag**2 > 4:
                        break
                    num_iter += 1
                col = int(TEXTURE_SIZE * num_iter / max_iter)
                screen_array[j, i] = TEXTURE_ARRAY[col, col]
        return screen_array

    def check_user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                for key in self.keys:
                    if key == event.key:
                        self.key_pressed[key] = True
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            if event.type == pygame.KEYUP:
                for key in self.keys:
                    if key == event.key:
                        self.key_pressed[key] = False

    @property
    def generate_frame(self):
        min_x = -WIDTH // 2 + self.offset[0]
        max_x = WIDTH // 2 + self.offset[0]
        min_y = -HEIGHT // 2 + self.offset[1]
        max_y = HEIGHT // 2 + self.offset[1]
        mid_x = (max_x + min_x) / 2
        mid_y = (max_y + min_y) / 2
        x = np.linspace(min_x, max_x, WIDTH)
        y = np.linspace(min_y, max_y, HEIGHT)
        x -= mid_x
        x *= self.koef
        x += mid_x
        y -= mid_y
        y *= self.koef
        y += mid_y
        return x, y

    def calculate_set(self):
        def zoom(koef):
            self.koef *= koef
            self.dx = DX * self.koef

        if self.key_pressed[pygame.K_UP]:
            self.offset[0] -= self.dx
        if self.key_pressed[pygame.K_DOWN]:
            self.offset[0] += self.dx
        if self.key_pressed[pygame.K_LEFT]:
            self.offset[1] -= self.dx
        if self.key_pressed[pygame.K_RIGHT]:
            self.offset[1] += self.dx
        if self.key_pressed[pygame.K_p]:
            zoom(0.9)
        if self.key_pressed[pygame.K_m]:
            zoom(1/0.9)
        if self.key_pressed[pygame.K_z]:
            self.c += CX * self.koef
        if self.key_pressed[pygame.K_x]:
            self.c += 1j * CX * self.koef
        if self.key_pressed[pygame.K_a]:
            self.c -= CX * self.koef
        if self.key_pressed[pygame.K_s]:
            self.c -= 1j * CX * self.koef

        """
        SOLUTION VIA NUMBA
        """
        x, y = self.generate_frame
        z = Plot.calculate_fractal(screen_array=self.plain_surface, X=x, Y=y, c=self.c, max_iter=100)
        return z

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
    fun = lambda z, c: np.array([z[0]**2 - z[1]**2 + c[0], 2 * z[0] * z[1] + c[1]])
    plot = Plot(fun)
    plot.draw()
