from enum import Enum
from itertools import cycle

import pygame

from ..utils import GameConfig, clamp
from .entity import Entity
from .floor import Floor
from .pipe import Pipe, Pipes


class PlayerMode(Enum):
    SHM = "SHM"
    NORMAL = "NORMAL"
    CRASH = "CRASH"


class Player(Entity):
    def __init__(self, config: GameConfig) -> None:
        image = config.images.player[0]
        x = int(config.window.width * 0.2)
        y = int((config.window.height - image.get_height()) / 2)
        super().__init__(config, image, x, y)
        self.min_y = -2 * self.h
        self.max_y = config.window.viewport_height - self.h * 0.75
        self.img_idx = 0
        self.img_gen = cycle([0, 1, 2, 1])
        self.frame = 0
        self.crashed = False
        self.crash_entity = None
        self.set_mode(PlayerMode.SHM)

    def set_mode(self, mode: PlayerMode) -> None:
        self.mode = mode
        if mode == PlayerMode.NORMAL:
            self.reset_vals_normal()
            self.config.sounds.wing.play()
        elif mode == PlayerMode.SHM:
            self.reset_vals_shm()
        elif mode == PlayerMode.CRASH:
            self.stop_wings()
            self.config.sounds.hit.play()
            if self.crash_entity == "pipe":
                self.config.sounds.die.play()
            self.reset_vals_crash()

    def reset_vals_normal(self) -> None:
        self.vel_y = -9  # player's velocity along Y axis
        self.max_vel_y = 10  # max vel along Y, max descend speed
        self.min_vel_y = -8  # min vel along Y, max ascend speed
        self.acc_y = 1  # players downward acceleration

        self.rot = 80  # player's current rotation
        self.vel_rot = -3  # player's rotation speed
        self.rot_min = -90  # player's min rotation angle
        self.rot_max = 20  # player's max rotation angle

        self.flap_acc = -9  # players speed on flapping
        self.flapped = False  # True when player flaps

    def reset_vals_shm(self) -> None:
        self.vel_y = 1  # player's velocity along Y axis
        self.max_vel_y = 4  # max vel along Y, max descend speed
        self.min_vel_y = -4  # min vel along Y, max ascend speed
        self.acc_y = 0.5  # players downward acceleration

        self.rot = 0  # player's current rotation
        self.vel_rot = 0  # player's rotation speed
        self.rot_min = 0  # player's min rotation angle
        self.rot_max = 0  # player's max rotation angle

        self.flap_acc = 0  # players speed on flapping
        self.flapped = False  # True when player flaps

    def reset_vals_crash(self) -> None:
        self.acc_y = 2
        self.vel_y = 7
        self.max_vel_y = 15
        self.vel_rot = -8

    def update_image(self):
        self.frame += 1
        if self.frame % 5 == 0:
            self.img_idx = next(self.img_gen)
            self.image = self.config.images.player[self.img_idx]
            self.w = self.image.get_width()
            self.h = self.image.get_height()

    def tick_shm(self) -> None:
        if self.vel_y >= self.max_vel_y or self.vel_y <= self.min_vel_y:
            self.acc_y *= -1
        self.vel_y += self.acc_y
        self.y += self.vel_y

    def tick_normal(self) -> None:
        if self.vel_y < self.max_vel_y and not self.flapped:
            self.vel_y += self.acc_y
        if self.flapped:
            self.flapped = False

        self.y = clamp(self.y + self.vel_y, self.min_y, self.max_y)
        self.rotate()

    def tick_crash(self) -> None:
        if self.min_y <= self.y <= self.max_y:
            self.y = clamp(self.y + self.vel_y, self.min_y, self.max_y)
            # rotate only when it's a pipe crash and bird is still falling
            if self.crash_entity != "floor":
                self.rotate()

        # player velocity change
        if self.vel_y < self.max_vel_y:
            self.vel_y += self.acc_y

    def rotate(self) -> None:
        self.rot = clamp(self.rot + self.vel_rot, self.rot_min, self.rot_max)

    def draw(self) -> None:
        self.update_image()
        if self.mode == PlayerMode.SHM:
            self.tick_shm()
        elif self.mode == PlayerMode.NORMAL:
            self.tick_normal()
        elif self.mode == PlayerMode.CRASH:
            self.tick_crash()

        self.draw_player()

    def draw_player(self) -> None:
        rotated_image = pygame.transform.rotate(self.image, self.rot)
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        self.config.screen.blit(rotated_image, rotated_rect)

    def stop_wings(self) -> None:
        self.img_gen = cycle([self.img_idx])

    def flap(self) -> None:
        if self.y > self.min_y:
            self.vel_y = self.flap_acc
            self.flapped = True
            self.rot = 80
            self.config.sounds.wing.play()

    def crossed(self, pipe: Pipe) -> bool:
        return pipe.cx <= self.cx < pipe.cx - pipe.vel_x

    def collided(self, pipes: Pipes, floor: Floor) -> bool:
        """returns True if player collides with floor or pipes."""

        # if player crashes into ground
        if self.collide(floor):
            self.crashed = True
            self.crash_entity = "floor"
            return True

        for pipe in pipes.upper:
            if self.collide(pipe):
                self.crashed = True
                self.crash_entity = "pipe"
                return True
        for pipe in pipes.lower:
            if self.collide(pipe):
                self.crashed = True
                self.crash_entity = "pipe"
                return True

        return False
    

    def get_gap_data(self, pipes):
        """Lấy dx, dy, v - khoảng cách đến tâm lỗ và vận tốc chim"""
        if not pipes.upper:
            return 0, 0, self.vel_y
        
        # Tìm ống gần nhất chưa vượt qua HOÀN TOÀN
        nearest_pipe = None
        for pipe in pipes.upper:
            # Thay đổi điều kiện: chỉ khi chim vượt qua hoàn toàn cột mới tìm cột tiếp theo
            if pipe.x + pipe.w > self.cx:  # Dùng cx thay vì x
                nearest_pipe = pipe
                break
        
        if not nearest_pipe:
            return 0, 0, self.vel_y
        
        # Tính tâm lỗ
        gap_center_x = nearest_pipe.x + nearest_pipe.w / 2
        gap_center_y = nearest_pipe.y + nearest_pipe.h + pipes.pipe_gap / 2
        
        # Tính khoảng cách
        dx = gap_center_x - self.cx  # Khoảng cách ngang
        dy = gap_center_y - self.cy  # Khoảng cách dọc  
        v = self.vel_y               # Vận tốc dọc của chim
        
        return dx, dy, v

    def get_detailed_state(self, pipes, floor):
        """Lấy trạng thái chi tiết cho AI/ML"""
        dx, dy, v = self.get_gap_data(pipes)
        
        return {
            'dx': dx,           # Khoảng cách ngang đến tâm lỗ
            'dy': dy,           # Khoảng cách dọc đến tâm lỗ
            'v': v,             # Vận tốc dọc chim
            'bird_y': self.y,   # Vị trí y của chim
            'bird_x': self.x,   # Vị trí x của chim
            'floor_y': floor.y, # Vị trí sàn
        }
