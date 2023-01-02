import pygame
import numpy as np
from numba import njit


DOT_COUNT = 80000
DX = 1e-3
FPS = 60
BACK_GROUND = (200, 200, 200)
GRAPH_COLOR = (10, 10, 10)
WIDTH = 800
HEIGHT = 800


class Plot:
    current_letter = "M"
    current_digit = "1"
    pointer = 0
    keys = [
        pygame.K_UP,
        pygame.K_DOWN,
        pygame.K_m,
        pygame.K_c,
        pygame.K_1,
        pygame.K_2,
        pygame.K_3,
        pygame.K_4,
    ]
    key_pressed = {key: False for key in keys}

    M1 = np.array([
        [0.00, 0.00],
        [0.00, 0.16],
    ])
    M2 = np.array([
        [0.85, 0.04],
        [-0.04, 0.85],
    ])
    M3 = np.array([
        [0.20, -0.26],
        [0.23, 0.22],
    ])
    M4 = np.array([
        [-0.15, 0.28],
        [0.26, 0.24],
    ])

    C1 = np.array([0.0, 0.0])
    C2 = np.array([0.00, 1.60])
    C3 = np.array([0.00, 1.60])
    C4 = np.array([0.00, 0.44])

    p = np.array([0.01, 0.85, 0.07, 0.07])

    def __init__(self, n):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.rule_sequence = np.array([np.random.choice([i for i in range(len(self.p))], p=self.p) for _ in range(n)])
        self.start_dot = np.array([0.0, 0.0])

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
                if event.key == pygame.K_LEFT:
                    self.pointer -= 1 if self.pointer > 0 else 0
                if event.key == pygame.K_RIGHT:
                    self.pointer += 1 if self.pointer < 3 and self.current_letter == "M" or self.pointer < 1 else 0
            if event.type == pygame.KEYUP:
                for key in self.keys:
                    if key == event.key:
                        self.key_pressed[key] = False

    def check_key_pressed(self):
        def change_scalar(koef):
            attr_name = self.current_letter + self.current_digit
            attr = getattr(self, attr_name)
            attr = attr.reshape(4) if attr.shape == (2, 2) else attr
            attr[self.pointer] += koef
            attr = attr.reshape((2, 2)) if len(attr) == 4 else attr
            setattr(self, attr_name, attr)

        if self.key_pressed[pygame.K_m]:
            self.current_letter = "M"
        if self.key_pressed[pygame.K_c]:
            self.pointer = 1 if self.pointer > 1 else self.pointer
            self.current_letter = "C"
        if self.key_pressed[pygame.K_1]:
            self.current_digit = "1"
        if self.key_pressed[pygame.K_2]:
            self.current_digit = "2"
        if self.key_pressed[pygame.K_3]:
            self.current_digit = "3"
        if self.key_pressed[pygame.K_4]:
            self.current_digit = "4"
        if self.key_pressed[pygame.K_UP]:
            change_scalar(DX)
        if self.key_pressed[pygame.K_DOWN]:
            change_scalar(-DX)


    def calculate_set(self):
        self.check_key_pressed()
        dots = Plot.calculate_set_numba(
            start_dot=self.start_dot,
            rule_sequence=self.rule_sequence,
            M=np.array([self.M1, self.M2, self.M3, self.M4]),
            C=np.array([self.C1, self.C2, self.C3, self.C4]),
            n=len(self.rule_sequence),
        )
        surface = np.full((HEIGHT, WIDTH), 255)
        dots *= 50
        dots += 200
        dots[:, 0] += 200
        dots = dots.astype(int)
        for dot in dots:
            surface[HEIGHT - int(dot[1]), WIDTH - int(dot[0])] = 0
        surface = np.rot90(surface)
        return surface

    @staticmethod
    @njit(fastmath=True)
    def calculate_set_numba(start_dot, rule_sequence, M, C, n):
        dots = np.zeros((n+1, 2))
        dots[0] = start_dot
        for i, rule_key in enumerate(rule_sequence):
            rule = lambda x: np.array([
                M[rule_key, 0, 0]*x[0] + M[rule_key, 0, 1]*x[1],
                M[rule_key, 1, 0]*x[0] + M[rule_key, 1, 1]*x[1]]
            ) + C[rule_key]
            dots[i+1] = rule(dots[i])
        return dots


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
            fps = str(round(self.clock.get_fps(), 2))
            val = getattr(self, self.current_letter + self.current_digit)
            val = list(val.reshape(4)) if val.shape == (2, 2) else list(val)
            val[self.pointer] = f"*{val[self.pointer]}*"
            pygame.display.set_caption(
                f"FPS={fps}{(5 - len(fps)) * '0'}; {self.current_letter + self.current_digit}="
                f"{val}"
            )


if __name__ == "__main__":
    plot = Plot(DOT_COUNT)
    plot.draw()
