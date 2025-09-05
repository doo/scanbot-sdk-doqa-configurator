from pathlib import Path

import cv2 as cv
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scanbotsdk


def plot_annotations(image_path: Path, annotations: scanbotsdk.DocumentQualityAnalyzerTrainingData):
    character_level_annotations = annotations.character_level_annotations

    img_array = cv.imread(str(image_path))
    cv.cvtColor(img_array, cv.COLOR_BGR2RGB, dst=img_array)  # Convert BGR to RGB for matplotlib

    def create_metric_plot(metric_name, metric_values_func, fig, ax):

        ax.imshow(img_array, alpha=0.8, cmap='gray')

        ax.set_xlim(0, img_array.shape[1])
        ax.set_ylim(img_array.shape[0], 0)  # Inverse y-axis for image coordinates

        all_values = [metric_values_func(char) for char in character_level_annotations.annotations]
        vmin, vmax = min(all_values), max(all_values)
        if metric_name == "Orientation":
            vmin, vmax = -180, 180
        elif metric_name == "Ocrability" or metric_name == "Contrast":
            vmin, vmax = 0, 1

        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if metric_name == "Orientation":
            cm_colors = np.vstack(
                [
                    plt.cm.RdYlGn(np.linspace(0, 1, plt.cm.RdYlGn.N)),
                    plt.cm.RdYlGn.reversed()(np.linspace(0, 1, plt.cm.RdYlGn.N)),
                ]
            )
            cmap = colors.ListedColormap(cm_colors)
        else:
            cmap = plt.cm.RdYlGn

        for char in character_level_annotations.annotations:
            value = metric_values_func(char)
            rect = patches.Rectangle(
                (
                    char.plot_center.x - char.plot_width / 2,
                    char.plot_center.y - char.plot_height / 2,
                ),
                char.plot_width,
                char.plot_height,
                linewidth=0.5,
                edgecolor='k',
                facecolor=cmap(norm(value)),
                alpha=0.6,
            )
            ax.add_patch(rect)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='bottom')
        cbar.set_label(f'{metric_name} Value')

        ax.set_title(f'Character-Level {metric_name} Visualization')

    cols = 5
    fig, [ax1a, ax2a, ax3a, ax4a, ax5a] = plt.subplots(1, cols, figsize=(6 * cols, 10))

    ax1a.imshow(img_array, alpha=1, cmap='gray')
    ax1a.set_title('Original Image')
    create_metric_plot('Font_Size', lambda char: char.font_size, fig, ax2a)
    create_metric_plot('Ocrability', lambda char: char.ocrability, fig, ax3a)
    create_metric_plot('Contrast', lambda char: char.contrast, fig, ax4a)
    create_metric_plot('Orientation', lambda char: char.orientation_normalized, fig, ax5a)

    plt.tight_layout()

    return plt
