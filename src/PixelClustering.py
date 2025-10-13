import numpy as np
import pandas as pd


class PixelClusteringTransformer:
    def __init__(self, n_pixel_clusters: int = 0):
        self.n_pixel_clusters = n_pixel_clusters

    def create_hist(self, gray_pixels) -> pd.Series:
        if self.n_pixel_clusters > 0:
            flat_img = gray_pixels.astype(np.uint8)
            hist = np.histogram(flat_img, bins=self.bin_boundaries())[0]
            hist = hist / len(flat_img)
            return pd.Series(hist, index=range(len(hist)))
        else:
            return pd.Series(dtype=float)

    def rgb_color_bin_representatives(self) -> list[list[float]]:
        if self.n_pixel_clusters == 0:
            return []
        bins = self.bin_boundaries()
        num_channels = 3
        return [[(bins[i] + bins[i + 1]) / 2 / 255] * num_channels for i in range(len(bins) - 1)]

    def export(self):
        return dict(
            n_pixel_clusters=self.n_pixel_clusters,
        )

    def bin_boundaries(self):
        assert self.n_pixel_clusters >= 3
        n = self.n_pixel_clusters - 1
        # We make the bins at the boundaries half as wide. This seems to improve the
        # ability of the model to detect too bright/dark images.
        return [
            0,
            255 / n / 2,
            *[255 / n / 2 + 255 / n * i for i in range(1, n - 1)],
            256 - 255 / n / 2,
            256,
        ]
