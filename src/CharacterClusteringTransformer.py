import pandas as pd
import scanbotsdk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_character_properties(
        character_level_annotations: scanbotsdk.CharacterLevelAnnotations,
        cluster_features: [str]
):
    results: [dict] = []

    for character_level_annotation in character_level_annotations.annotations:
        result = {}
        for feature_name in cluster_features:
            if feature_name == "Contrast":
                value = character_level_annotation.contrast
            elif feature_name == "Ocrability":
                value = character_level_annotation.ocrability
            elif feature_name == "FontSize":
                value = character_level_annotation.font_size
            else:
                raise ValueError(f"Unknown feature name: {feature_name}")
            result[feature_name] = value
        results.append(result)

    return results


class CharacterClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int, cluster_features: [str] = None):
        self.n_clusters = n_clusters
        self.cluster_features = cluster_features if cluster_features is not None else ["Contrast", "Ocrability",
                                                                                       "FontSize"]
        self.pipeline = None
        self.fitted_ = None  # Read by scikit-learn's check_is_fitted

    def fit(self, X, y=None):
        from sklearn.cluster import KMeans
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=self.n_clusters))
        ])

        character_properties = [
            char_properties
            for sample in X['character_level_annotations']
            for char_properties in get_character_properties(sample, self.cluster_features)
        ]

        self.pipeline.fit(pd.DataFrame(character_properties)[self.cluster_features])
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.pipeline is None or not self.fitted_:
            raise RuntimeError("You must fit the model before transforming data.")

        hists = []

        for sample in X['character_level_annotations']:
            df = pd.DataFrame(get_character_properties(sample, self.cluster_features))
            cluster_labels = self.pipeline.predict(df[self.cluster_features])
            hist = pd.Series(cluster_labels).value_counts(normalize=True).sort_index()
            hist = hist.reindex(range(self.n_clusters), fill_value=0.0)  # fill missing clusters with 0.0
            hists.append(hist)

        return hists

    def export(self):
        if self.pipeline is None or not self.fitted_:
            raise RuntimeError("You must fit the model before exporting.")

        kmeans_model = self.pipeline.named_steps['kmeans']
        cluster_centers = kmeans_model.cluster_centers_.tolist()
        normalizer = dict(
            mean=[self.pipeline.named_steps['scaler'].mean_[i] for i, feature in enumerate(self.cluster_features)],
            scale=[self.pipeline.named_steps['scaler'].scale_[i] for i, feature in enumerate(self.cluster_features)],
        )

        return dict(
            clustering_features=self.cluster_features,
            character_cluster_centers=cluster_centers,
            normalizer=normalizer
        )
