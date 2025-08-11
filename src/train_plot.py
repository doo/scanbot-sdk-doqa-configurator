from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import GridSearchCV


def plot_grid_search(clf: GridSearchCV, output_dir: Path):
    results = pd.DataFrame(clf.cv_results_)
    results['line_label'] = results.apply(
        lambda row: f"{row['param_clustering__cluster_features']},kernel={row['param_svm__kernel']},C={row['param_svm__C']},gamma_factor={row['param_svm__gamma_factor']}",
        axis=1,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            'CV Accuracy vs. Number of Clusters',
            'Number of Support Vectors vs. Number of Clusters',
        ),
        vertical_spacing=0.1,
        shared_xaxes=True,
    )

    for label, group in results.groupby('line_label'):
        fig.add_trace(
            go.Scatter(
                x=group['param_clustering__n_clusters'],
                y=group['mean_test_accuracy'],
                mode='lines+markers',
                name=label,
                legendgroup=label,
                hovertemplate=f"%{{x}} clusters<br>CV Accuracy: %{{y:.3f}}<br>{label}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    for label, group in results.groupby('line_label'):
        fig.add_trace(
            go.Scatter(
                x=group['param_clustering__n_clusters'],
                y=group['mean_test_num_support_vectors'],
                mode='lines+markers',
                name=label,
                legendgroup=label,
                showlegend=False,
                hovertemplate=f"%{{x}} clusters<br>Support Vectors: %{{y}}<br>{label}<extra></extra>",
                line=dict(dash='dash'),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=1200,
        width=1500,
        legend_title="Configuration",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    fig.update_xaxes(title_text='Number of Clusters', row=2, col=1)
    fig.update_yaxes(title_text='CV Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Number of Support Vectors', row=2, col=1)

    plot_path = output_dir / 'cv_accuracy_vs_clusters.html'
    print(f"Saving plot to {plot_path}")
    fig.write_html(str(plot_path))


def plot_stacked_area_uncertainty(
    y: pd.Series,
    y_pred: pd.Series,
    output_dir: Path,
    threshold_midpoints: list[float],
    subtitle: str = "",
):
    thresholds = np.linspace(0, 0.5, 51)
    total_samples = len(y)

    tp_counts = []
    fp_counts = []
    up_counts = []
    un_counts = []
    tn_counts = []
    fn_counts = []

    for threshold in thresholds:
        upper_bound = 0.5 + threshold
        lower_bound = 0.5 - threshold

        tp = ((y == 1) & (y_pred >= upper_bound)).sum()
        fp = ((y == 0) & (y_pred >= upper_bound)).sum()
        up = ((y == 1) & (y_pred > lower_bound) & (y_pred < upper_bound)).sum()
        un = ((y == 0) & (y_pred > lower_bound) & (y_pred < upper_bound)).sum()
        tn = ((y == 0) & (y_pred <= lower_bound)).sum()
        fn = ((y == 1) & (y_pred <= lower_bound)).sum()

        tp_counts.append(tp / total_samples * 100)
        fp_counts.append(fp / total_samples * 100)
        up_counts.append(up / total_samples * 100)
        un_counts.append(un / total_samples * 100)
        tn_counts.append(tn / total_samples * 100)
        fn_counts.append(fn / total_samples * 100)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=fn_counts,
            fill='tonexty',
            mode='none',
            name='False Negatives',
            fillcolor='rgba(255, 0, 0, 0.7)',
            stackgroup='one',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=tp_counts,
            fill='tonexty',
            mode='none',
            name='True Positives',
            fillcolor='rgba(0, 128, 0, 0.7)',
            stackgroup='one',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=up_counts,
            fill='tonexty',
            mode='none',
            name='Uncertain Positives',
            fillcolor='rgba(0, 255, 0, 0.5)',
            stackgroup='one',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=un_counts,
            fill='tonexty',
            mode='none',
            name='Uncertain Negatives',
            fillcolor='rgba(255, 165, 0, 0.5)',
            stackgroup='one',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=tn_counts,
            fill='tonexty',
            mode='none',
            name='True Negatives',
            fillcolor='rgba(0, 0, 255, 0.7)',
            stackgroup='one',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=fp_counts,
            fill='tonexty',
            mode='none',
            name='False Positives',
            fillcolor='rgba(128, 0, 128, 0.7)',
            stackgroup='one',
        )
    )

    fig.update_layout(
        title='Classification Uncertainty Analysis; ' + subtitle,
        xaxis_title='Uncertainty Threshold',
        yaxis_title='Percentage of Samples',
        hovermode='x unified',
        width=1000,
        height=600,
    )

    plot_path = output_dir / 'uncertainty_analysis.html'
    fig.write_html(str(plot_path))
    print(f"Uncertainty analysis plot saved to {plot_path}")
