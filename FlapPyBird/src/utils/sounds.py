import sys

import pygame
from .paths import audio_path


class Sounds:
    die: pygame.mixer.Sound
    hit: pygame.mixer.Sound
    point: pygame.mixer.Sound
    swoosh: pygame.mixer.Sound
    wing: pygame.mixer.Sound

    def __init__(self) -> None:
        if "win" in sys.platform:
            ext = "wav"
        else:
            ext = "ogg"

        self.die = pygame.mixer.Sound(audio_path(f"die.{ext}"))
        self.hit = pygame.mixer.Sound(audio_path(f"hit.{ext}"))
        self.point = pygame.mixer.Sound(audio_path(f"point.{ext}"))
        self.swoosh = pygame.mixer.Sound(audio_path(f"swoosh.{ext}"))
        self.wing = pygame.mixer.Sound(audio_path(f"wing.{ext}"))
