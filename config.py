import os
from enum import Enum


class AutoLabelingMode(Enum):
    LOCAL = "LOCAL"
    AWS = "AWS"
    DEBUG = "DEBUG"

    @classmethod
    def get_mode_from_user_input(cls, s: str):
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.LOCAL


def set_autolabeling_mode():
    run_env = os.getenv("AUTOLABELING_MODE", "LOCAL")
    return AutoLabelingMode.get_mode_from_user_input(run_env)
