import numpy as np
import pandas as pd


class PixelClusteringTransformer:
    def __init__(self, n_pixel_clusters: int = 0):
        self.n_pixel_clusters = n_pixel_clusters

    def create_hist(self, gray_pixels) -> pd.Series:
        if self.n_pixel_clusters > 0:
            img = gray_pixels.astype(np.uint8)
            hist = np.histogram(img, bins=self.bins())[0]
            hist = hist / len(img)
            return pd.Series(hist, index=range(len(hist)))
        else:
            return pd.Series(dtype=float)

    def rgb_color_bin_representatives(self) -> list[list[float]]:
        if self.n_pixel_clusters == 0:
            return []
        bins = self.bins()
        return [[(bins[i] + bins[i + 1]) / 2 / 255] * 3 for i in range(len(bins) - 1)]

    def export(self):
        return dict(
            n_pixel_clusters=self.n_pixel_clusters,
        )

    def bins(self):
        # We make the bins at the boundaries half as wide. This seems to improve the
        # ability of the model to detect too bright/dark images.
        return [
            0,
            255 / self.n_pixel_clusters / 2,
            *[
                255 / self.n_pixel_clusters / 2 + 255 / self.n_pixel_clusters * i
                for i in range(1, self.n_pixel_clusters - 1)
            ],
            256 - 255 / self.n_pixel_clusters / 2,
            256,
        ]
