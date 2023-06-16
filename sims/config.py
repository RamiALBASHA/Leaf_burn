from json import load

from pathlib import Path


class Pot:
    def __init__(self):
        self.height = 23  # cm
        self.radius_upper = 13 / 2.  # cm
        self.radius_lower = 10 / 2.  # cm
        self.color = (0, 0, 0)


class Config:
    def __init__(self):
        self.path_root = Path(__file__).parent
        self.path_digit = self.path_root / 'source/digit'
        self.path_weather = self.path_root / 'source/weather.csv'
        self.path_preprocessed_data = self.path_root / 'preprocessed_data'

        self.unit_digit: str = 'cm'
        self.spacing_between_rows: float = 100  # cm
        self.spacing_on_row: float = Pot().radius_upper * 2.  # cm
        self.row_angle_with_sout: float = 140
        self.leaf_height_attribute: str = 'TopPosition'
        self.extinction_coefficient_wind: float = 0.5

        with open(self.path_root / 'params.json', mode='r') as f:
            self.params = load(f)
        self.params['planting']['spacing_on_row'] = self.spacing_on_row / 100.
        self.params['planting']['spacing_between_rows'] = self.spacing_between_rows / 100.
