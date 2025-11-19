# FlapPyBird/src/utils/paths.py
import os

# Thư mục hiện tại là utils/, nên src_dir = ../
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FlapPyBird/ là thư mục cha của src/
FLAPPYBIRD_DIR = os.path.dirname(SRC_DIR)
# assets/ nằm cùng cấp với src/
ASSETS_DIR = os.path.join(FLAPPYBIRD_DIR, "assets")
SPRITES_DIR = os.path.join(ASSETS_DIR, "sprites")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")


def sprite_path(name: str) -> str:
    """Trả về đường dẫn đầy đủ đến file sprite."""
    return os.path.join(SPRITES_DIR, name)


def audio_path(name: str) -> str:
    """Trả về đường dẫn đầy đủ đến file âm thanh."""
    return os.path.join(AUDIO_DIR, name)
