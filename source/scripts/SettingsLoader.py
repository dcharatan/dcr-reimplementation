import json
import numpy as np


class SettingsLoader:
    SETTINGS_DEFAULTS = {
        "scene": "data/blender-scenes/forest.blend",
        "image_shape": [1000, 1160, 3],
        "reference_location": [20, -20, 18],
        "reference_target": [0, 0, 0],
        "initial_location": [19, -19, 17],
        "initial_target": [1, -1, 1],
        "save_folder": "results/forest",
        "hand_eye_euler_xyz": [3, 3, 3],
        "hand_eye_translation": [0.2, 0.1, 0.4],
        "s_initial": 2,
        "s_min": 0.05,
    }

    @staticmethod
    def load_settings(file_name: str):
        with open(file_name, "r") as f:
            settings = json.load(f)
        SettingsLoader.validate_settings(settings)

        # Use the defaults for unspecified settings.
        settings = {**SettingsLoader.SETTINGS_DEFAULTS, **settings}

        # Convert arrays to numpy.
        for key, value in settings.items():
            if isinstance(value, list):
                settings[key] = np.array(value, dtype=np.float64)

        return settings

    @staticmethod
    def validate_settings(settings: dict):
        for key in settings:
            if key not in SettingsLoader.SETTINGS_DEFAULTS:
                raise Exception(f'Unknown setting "{key}"')
