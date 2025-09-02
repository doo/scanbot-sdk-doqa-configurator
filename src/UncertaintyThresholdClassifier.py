import pandas as pd
from configurator_utils import ThresholdWaypoints


class UncertaintyThresholdClassifier:
    def __init__(self, threshold_waypoints: ThresholdWaypoints, thresholds: list[float]):
        self.threshold_waypoints = threshold_waypoints
        self.thresholds = thresholds

    def predict(self, X: pd.Series) -> pd.Series:
        upper_bound = self.linear_interpolate_threshold(1)
        lower_bound = self.linear_interpolate_threshold(0)
        return X.apply(lambda x: 1 if x >= upper_bound else 0 if x <= lower_bound else -1)

    def linear_interpolate_threshold(self, label):
        at0p5 = self.threshold_waypoints["threshold_values_0p5"][label]
        at1p0 = self.threshold_waypoints["threshold_values_1p0"][label]

        if self.thresholds[label] < 0.5:
            return self.thresholds[label] * at0p5 * 2
        else:
            return (self.thresholds[label] - 0.5) * 2 * (at1p0 - at0p5) + at0p5
