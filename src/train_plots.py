from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from configurator_utils import ThresholdWaypoints
from plotly.subplots import make_subplots
from UncertaintyThresholdClassifier import UncertaintyThresholdClassifier


def plot_grid_search(results, output_dir: Path):
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


def plot_classification(
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

    correct_counts = [tp + tn for tp, tn in zip(tp_counts, tn_counts)]
    incorrect_counts = [fp + fn for fp, fn in zip(fp_counts, fn_counts)]
    uncertain_counts = [up + un for up, un in zip(up_counts, un_counts)]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Classification results",
            "Classification performance",
            "",
            "Threshold Analysis",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"type": "table", "colspan": 2}, None],
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.4,
        row_heights=[0.7, 0.3],
    )
    # First subplot: Stacked area chart
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=tn_counts,
            fill='tonexty',
            mode='none',
            name='True Negatives ("unacceptable" document correctly classified)',
            fillcolor='rgba(0, 128, 0, 0.7)',
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
            fillcolor='rgba(160, 160, 0, 0.7)',
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
            fillcolor='rgba(128, 0, 0, 0.7)',
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
            fillcolor='rgba(230, 0, 0, 0.6)',
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
            fillcolor='rgba(230, 230, 0, 0.6)',
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
            fillcolor='rgba(0, 230, 0, 0.6)',
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
            y=uncertain_counts,
            mode='lines',
            name='% Uncertain',
            line=dict(color='orange', width=3),
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
            name='% Incorrect',
            line=dict(color='red', width=3),
            legendgroup='performance',
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=correct_counts,
            mode='lines',
            name='% Correct',
            line=dict(color='green', width=3),
            legendgroup='performance',
        ),
        row=1,
        col=2,
    )

    # Third subplot: Threshold analysis table
    sample_thresholds = [0, 0.3, 0.5, 0.7]
    table_data = []

    for threshold in sample_thresholds:
        idx = int(threshold * 100)  # Find closest index in thresholds array
        correct_pct = correct_counts[idx]
        uncertain_pct = uncertain_counts[idx]
        incorrect_pct = incorrect_counts[idx]

        table_data.append(
            {
                'Uncertainty Threshold': f"{threshold:.1f}",
                '% Correct': f"{correct_pct:.0f}%",
                '% Uncertain': f"{uncertain_pct:.0f}%",
                '% Incorrect': f"{incorrect_pct:.0f}%",
            }
        )

    table_df = pd.DataFrame(table_data)

    fig.add_trace(
        go.Table(
            header=dict(
                values=table_df.columns,
                fill_color='paleturquoise',
                align='center',
                font=dict(size=12),
            ),
            cells=dict(
                values=table_df.transpose(),
                fill_color='lavender',
                align='center',
                font=dict(size=11),
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        width=1200,
        height=800,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="top", y=0.45, xanchor="center", x=0.5),
    )

    fig.update_xaxes(title_text='Uncertainty Threshold', row=1, col=1)
    fig.update_xaxes(title_text='Uncertainty Threshold', row=1, col=2)
    fig.update_yaxes(title_text='% of Training Samples', row=1, col=1)
    fig.update_yaxes(title_text='% of Training Samples', row=1, col=2)

    if output_dir:
        html_plot_path = Path(output_dir) / 'uncertainty_analysis.html'
        fig.write_html(str(html_plot_path))
        print(f"Uncertainty analysis plot saved to {html_plot_path}")

    return fig
