class Pot:
    def __init__(self):
        self.height = 23  # cm
        self.radius_upper = 13 / 2.  # cm
        self.radius_lower = 10 / 2.  # cm
        self.color = (0, 0, 0)


class Config:
    def __init__(self):
        self.unit_digit: str = 'cm'
        self.spacing_between_rows: float = 100  # cm
        self.spacing_on_row: float = Pot().radius_upper * 2.  # cm
        self.row_angle_with_sout: float = 140
        self.leaf_height_attribute: str = 'TopPosition'
        self.extinction_coefficient_wind: float = 0.5
