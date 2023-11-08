import os
from enum import Enum


class AutoLabelingMode(Enum):
    LOCAL = "LOCAL"
    AWS = "AWS"
    DEBUG = "DEBUG"

    @staticmethod
    def get_mode_from_user_input(s: str):
        try:
            return AutoLabelingMode(s.upper())
        except KeyError:
            return AutoLabelingMode.LOCAL


def set_autolabeling_mode():
    run_env = os.getenv("AUTOLABELING_MODE", "LOCAL")
    return AutoLabelingMode.get_mode_from_user_input(run_env)
