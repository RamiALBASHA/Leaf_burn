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
        self.path_output_dir = self.path_root / 'output'

        self.unit_digit: str = 'cm'
        self.spacing_between_rows: float = 180  # cm
        self.spacing_on_row: float = Pot().radius_upper * 2.  # cm
        self.row_angle_with_sout: float = 140
        self.leaf_height_attribute: str = 'TopPosition'
        self.extinction_coefficient_wind: float = 0.5

        with open(self.path_root / 'params.json', mode='r') as f:
            self.params = load(f)
        self.params['planting']['spacing_on_row'] = self.spacing_on_row / 100.
        self.params['planting']['spacing_between_rows'] = self.spacing_between_rows / 100.
        self.scenarios = dict(
            intermediate={
                'd0': 7,
                'photo_inhibition': {
                    "dhd_inhib_beg": 200,
                    "dHd_inhib_max": 200,
                    "psi_inhib_beg": -1.5,
                    "psi_inhib_max": -3,
                    "temp_inhib_beg": 100,
                    "temp_inhib_max": 150}},
            biochemical_dominant={
                'd0': 7,
                "photo_inhibition": {
                    "dhd_inhib_beg": 195,
                    "dHd_inhib_max": 190,
                    "psi_inhib_beg": -1.5,
                    "psi_inhib_max": -3,
                    "temp_inhib_beg": 35,
                    "temp_inhib_max": 40}},
            stomatal_sensitivity_dominant={
                'd0': 1,
                'photo_inhibition': {
                    "dhd_inhib_beg": 200,
                    "dHd_inhib_max": 200,
                    "psi_inhib_beg": -1.5,
                    "psi_inhib_max": -3,
                    "temp_inhib_beg": 100,
                    "temp_inhib_max": 150}},
            stomatal_sensitivity_weak={
                'd0': 60,
                'photo_inhibition': {
                    "dhd_inhib_beg": 200,
                    "dHd_inhib_max": 200,
                    "psi_inhib_beg": -1.5,
                    "psi_inhib_max": -3,
                    "temp_inhib_beg": 100,
                    "temp_inhib_max": 150}}
        )
