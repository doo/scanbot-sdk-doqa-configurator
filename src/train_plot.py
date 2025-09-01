from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import GridSearchCV
from train_utils import ThresholdWaypoints
from UncertaintyThresholdClassifier import UncertaintyThresholdClassifier


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
    output_dir,
    threshold_waypoints: ThresholdWaypoints,
):
    thresholds = np.linspace(0, 1, 101)
    total_samples = len(y)

    tp_counts = []
    fp_counts = []
    up_counts = []
    un_counts = []
    tn_counts = []
    fn_counts = []

    for threshold in thresholds:
        classifier = UncertaintyThresholdClassifier(
            threshold_waypoints=threshold_waypoints, thresholds=[threshold, threshold]
        )
        y_pred_class = classifier.predict(y_pred)

        tp = ((y == 1) & (y_pred_class == 1)).sum()
        fp = ((y == 0) & (y_pred_class == 1)).sum()
        up = ((y == 1) & (y_pred_class == -1)).sum()
        un = ((y == 0) & (y_pred_class == -1)).sum()
        tn = ((y == 0) & (y_pred_class == 0)).sum()
        fn = ((y == 1) & (y_pred_class == 0)).sum()

        tp_counts.append(tp / total_samples * 100)
        fp_counts.append(fp / total_samples * 100)
        up_counts.append(up / total_samples * 100)
        un_counts.append(un / total_samples * 100)
        tn_counts.append(tn / total_samples * 100)
        fn_counts.append(fn / total_samples * 100)

    # Calculate performance metrics
    correct_counts = [tp + tn for tp, tn in zip(tp_counts, tn_counts)]
    incorrect_counts = [fp + fn for fp, fn in zip(fp_counts, fn_counts)]
    uncertain_counts = [up + un for up, un in zip(up_counts, un_counts)]

    # Create subplot figure
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Classification results", "Classification performance"),
        horizontal_spacing=0.1,
    )

    # First subplot: Original stacked area chart
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=tn_counts,
            fill='tonexty',
            mode='none',
            name='True Negatives ("unacceptable" document correctly classified)',
            fillcolor='rgba(0, 0, 255, 0.7)',
            stackgroup='one',
            legendgroup='classification',
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=fp_counts,
            fill='tonexty',
            mode='none',
            name='False Positives ("unacceptable" document classified as "acceptable")',
            fillcolor='rgba(128, 0, 128, 0.7)',
            stackgroup='one',
            legendgroup='classification',
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=un_counts,
            fill='tonexty',
            mode='none',
            name='Uncertain Negatives ("unacceptable" document classified as "uncertain")',
            fillcolor='rgba(255, 165, 0, 0.5)',
            stackgroup='one',
            legendgroup='classification',
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=up_counts,
            fill='tonexty',
            mode='none',
            name='Uncertain Positives ("acceptable" document classified as "uncertain")',
            fillcolor='rgba(0, 255, 0, 0.5)',
            stackgroup='one',
            legendgroup='classification',
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=fn_counts,
            fill='tonexty',
            mode='none',
            name='False Negatives ("acceptable" document classified as "unacceptable")',
            fillcolor='rgba(255, 0, 0, 0.7)',
            stackgroup='one',
            legendgroup='classification',
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=tp_counts,
            fill='tonexty',
            mode='none',
            name='True Positives ("acceptable" document correctly classified)',
            fillcolor='rgba(0, 128, 0, 0.7)',
            stackgroup='one',
            legendgroup='classification',
        ),
        row=1,
        col=1,
    )

    # Second subplot: Performance metrics
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=correct_counts,
            mode='lines',
            name='Correctly labeled samples',
            line=dict(color='green', width=3),
            legendgroup='performance',
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=incorrect_counts,
            mode='lines',
            name='Incorrectly labeled samples',
            line=dict(color='red', width=3),
            legendgroup='performance',
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=uncertain_counts,
            mode='lines',
            name='Uncertain labeled samples',
            line=dict(color='orange', width=3),
            legendgroup='performance',
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        width=1200,
        height=700,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
    )

    fig.update_xaxes(title_text='Uncertainty Threshold', row=1, col=1)
    fig.update_xaxes(title_text='Uncertainty Threshold', row=1, col=2)
    fig.update_yaxes(title_text='Percentage of Training Samples (%)', row=1, col=1)
    fig.update_yaxes(title_text='Percentage of Training Samples (%)', row=1, col=2)

    if output_dir:
        html_plot_path = Path(output_dir) / 'uncertainty_analysis.html'
        fig.write_html(str(html_plot_path))
        fig.write_image(str(Path(output_dir) / 'uncertainty_analysis.svg'), width=1200, height=700)
        print(f"Uncertainty analysis plot saved to {html_plot_path}")

    return fig
