import pygame
import numpy as np
from numba import njit


FPS = 60
BACK_GROUND = (200, 200, 200)
GRAPH_COLOR = (10, 10, 10)
WIDTH = 900
HEIGHT = 900
DX = 10


@njit(fastmath=True, parallel=True)
def calculate_fractal(z, c, scale, offset, center, n):
    x, y = z[0], z[1]
    for _ in range(n):
        new_x = x * x - y * y + c[0]
        new_y = 2 * x * y + c[1]
        x, y = new_x, new_y
    new_z = np.sqrt(x**2 + y**2)
    return new_z


class Plot:
    keys = [
        pygame.K_UP,
        pygame.K_DOWN,
        pygame.K_LEFT,
        pygame.K_RIGHT,
        pygame.K_p,
        pygame.K_m
    ]
    key_pressed = {key: False for key in keys}

    def __init__(self, fun):
        pygame.init()
        self.fun = fun
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.surface = np.full((HEIGHT, WIDTH), 255)
        self.offset = np.array([0, 0], dtype=float)
        self.center = np.array([0, 0])
        self.c = np.array([
            0.363,
            0.363,
            # -0.65
        ])
        self.z = np.zeros((2, WIDTH, HEIGHT), dtype=float)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                self.z[0, y, x] = x
                self.z[1, y, x] = y
        self.z[0] -= WIDTH // 2
        self.z[1] -= HEIGHT // 2
        # self.surface[0] -= WIDTH // 2
        # self.surface[1] -= HEIGHT // 2
        self.koef = 3e-3
        self.dx = DX * self.koef
        self.abs = lambda x: np.sqrt(x[0, :, :]**2 + x[1, :, :]**2)

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
        x = np.linspace(
            [min_x for _ in range(WIDTH)],
            [max_x for _ in range(WIDTH)],
            WIDTH
        ).T
        y = np.linspace(
            [min_y for _ in range(HEIGHT)],
            [max_y for _ in range(HEIGHT)],
            HEIGHT
        )
        x -= mid_x
        x *= self.koef
        x += mid_x
        y -= mid_y
        y *= self.koef
        y += mid_y
        return np.stack([x, y])

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

        z = self.generate_frame
        z = calculate_fractal(
            z=z, c=self.c, scale=self.koef,
            offset=self.offset, center=self.center, n=100,
        )
        z[z < 1] = 1
        return np.stack([self.surface * z] * 3, axis=2)

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
