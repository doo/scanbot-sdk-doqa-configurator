import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CharacterClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int = 0, cluster_features: list[str] = []):
        self.n_clusters = n_clusters
        self.cluster_features = (
            cluster_features
            if cluster_features is not None
            else ["Contrast", "Ocrability", "FontSize"]
        )
        self.pipeline = None
        self.fitted_ = None  # Read by scikit-learn's check_is_fitted

    def fit(self, X, y=None):
        from sklearn.cluster import KMeans

        self.pipeline = Pipeline(
            [('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=self.n_clusters))]
        )

        character_properties = pd.concat(
            [sample[self.cluster_features] for sample in X['character_level_annotations']]
        )

        self.pipeline.fit(character_properties)
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.pipeline is None or not self.fitted_:
            raise RuntimeError("You must fit the model before transforming data.")

        hists = []

        for sample in X['character_level_annotations']:
            df = sample[self.cluster_features]
            cluster_labels = self.pipeline.predict(df[self.cluster_features])
            hist = pd.Series(cluster_labels).value_counts(normalize=True).sort_index()
            hist = hist.reindex(
                range(self.n_clusters), fill_value=0.0
            )  # fill missing clusters with 0.0
            hists.append(hist)

        return hists

    def export(self):
        if self.pipeline is None or not self.fitted_:
            raise RuntimeError("You must fit the model before exporting.")

        kmeans_model = self.pipeline.named_steps['kmeans']
        cluster_centers = kmeans_model.cluster_centers_.tolist()
        normalizer = dict(
            mean=[
                self.pipeline.named_steps['scaler'].mean_[i]
                for i, feature in enumerate(self.cluster_features)
            ],
            scale=[
                self.pipeline.named_steps['scaler'].scale_[i]
                for i, feature in enumerate(self.cluster_features)
            ],
        )

        return dict(
            clustering_features=self.cluster_features,
            character_cluster_centers=cluster_centers,
            normalizer=normalizer,
        )
