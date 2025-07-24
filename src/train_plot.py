from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import GridSearchCV


def plot_grid_search(clf: GridSearchCV, output_dir: Path):
    results = pd.DataFrame(clf.cv_results_)
    results['line_label'] = results.apply(
        lambda
            row: f"{row['param_clustering__cluster_features']},kernel={row['param_svm__kernel']},C={row['param_svm__C']},gamma_factor={row['param_svm__gamma_factor']}",
        axis=1
    )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('CV Accuracy vs. Number of Clusters', 'Number of Support Vectors vs. Number of Clusters'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    for label, group in results.groupby('line_label'):
        fig.add_trace(go.Scatter(
            x=group['param_clustering__n_clusters'],
            y=group['mean_test_accuracy'],
            mode='lines+markers',
            name=label,
            legendgroup=label,
            hovertemplate=f"%{{x}} clusters<br>CV Accuracy: %{{y:.3f}}<br>{label}<extra></extra>"
        ), row=1, col=1)

    for label, group in results.groupby('line_label'):
        fig.add_trace(go.Scatter(
            x=group['param_clustering__n_clusters'],
            y=group['mean_test_num_support_vectors'],
            mode='lines+markers',
            name=label,
            legendgroup=label,
            showlegend=False,
            hovertemplate=f"%{{x}} clusters<br>Support Vectors: %{{y}}<br>{label}<extra></extra>",
            line=dict(dash='dash')
        ), row=2, col=1)

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