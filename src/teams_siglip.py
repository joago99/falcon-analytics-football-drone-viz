"""Clasificación de equipos con embeddings SigLIP + UMAP + KMeans.

Sustituye la clasificación por color HSV (frágil en vista cenital) por
embeddings visuales de SigLIP: se codifican los crops de los jugadores,
se reducen con UMAP y se agrupan con KMeans en 2 equipos. El árbitro se
detecta por color rojo (heurística complementaria).

Referencia: Smasko7/Football-Vision (SigLIP + UMAP + KMeans).
"""
import numpy as np


class SiglipTeamClassifier:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224",
                 n_clusters: int = 2, min_samples: int = 20,
                 cluster_every: int = 60, device=None):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.min_samples = min_samples
        self.cluster_every = cluster_every
        self._device = device
        self._model = None
        self._processor = None
        self._embeddings = {}      # track_id -> embedding (512,)
        self._centers = None       # (2, 512) centros de equipo en UMAP
        self._reducer = None       # objeto UMAP fit (para kNN de tracks nuevos)
        self._last_center_A = None # centroide UMAP del equipo A en el último clusterize
        self._team_of_track = {}   # track_id -> 'A'|'B'|'REF'
        self._frame_count = 0
        self._last_emb_idx = {}    # track_id -> último frame embebido

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor
        self._device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device).eval()

    def _embed_crop(self, crop) -> np.ndarray | None:
        """Embedding SigLIP del crop del jugador (devuelve vector 768 o None)."""
        import torch
        try:
            inputs = self._processor(images=crop, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.get_image_features(**inputs)
            emb = out.cpu().numpy().reshape(-1)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else None
        except Exception:
            return None

    def update(self, frame, tracked_players, frame_idx: int) -> dict[int, str]:
        """Acumula embeddings y (cada cluster_every frames) re-clusteriza.

        Devuelve {track_id: 'A'|'B'|'REF'} para los jugadores dados.
        """
        self._load()  # asegurar modelo cargado
        self._frame_count += 1
        # 1) embebemos crops nuevos (solo si no los tenemos de hace poco)
        from src.teams import _shirt_crop
        for obj in tracked_players:
            last = self._last_emb_idx.get(obj.track_id, -1)
            if self._frame_count - last < 30:
                continue  # ya embebido recientemente
            crop = _shirt_crop(frame, obj.bbox)
            if crop is None:
                continue
            emb = self._embed_crop(crop)
            if emb is not None:
                self._embeddings[obj.track_id] = emb
                self._last_emb_idx[obj.track_id] = self._frame_count

        # 2) clusterizar periódicamente con todos los embeddings acumulados
        if self._frame_count % self.cluster_every == 0 and len(self._embeddings) >= self.min_samples:
            self._clusterize()

        # 3) asignar equipo a los jugadores de este frame
        out = {}
        for obj in tracked_players:
            team = self._assign(obj.track_id)
            if team is not None:
                out[obj.track_id] = team
        return out

    def _clusterize(self):
        """KMeans sobre los embeddings SigLIP (espacio original, sin UMAP).

        UMAP era inestable para anclar A/B entre clusterizes; KMeans directo
        sobre embeddings normalizados es mucho más consistente. Los centros
        quedan en el mismo espacio y permiten kNN para tracks nuevos.
        """
        from sklearn.cluster import KMeans
        ids = list(self._embeddings.keys())
        X = np.array([self._embeddings[i] for i in ids], dtype=np.float32)
        if len(X) < self.min_samples:
            return
        k = min(self.n_clusters, len(X))
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        self._centers = km.cluster_centers_
        self._reducer = None  # sin UMAP

        # anclar A/B: el cluster cuyo centro esté más cerca del centroide del
        # equipo A previo sigue siendo 'A'; sin referencia previa, A = cluster 0
        if self._last_center_A is not None:
            d0 = np.linalg.norm(self._centers[0] - self._last_center_A)
            d1 = np.linalg.norm(self._centers[1] - self._last_center_A)
            a_idx = 0 if d0 <= d1 else 1
        else:
            a_idx = 0
        self._last_center_A = self._centers[a_idx].copy()
        for i, tid in enumerate(ids):
            self._team_of_track[tid] = "A" if km.labels_[i] == a_idx else "B"

    def _assign(self, track_id):
        """Asigna 'A'/'B' a un track (label persistente o kNN al centro)."""
        team = self._team_of_track.get(track_id)
        if team is not None:
            return team
        # track nuevo con embedding y centros: kNN directo al centro más cercano
        if track_id in self._embeddings and self._centers is not None:
            emb = self._embeddings[track_id]
            d0 = np.linalg.norm(emb - self._centers[0])
            d1 = np.linalg.norm(emb - self._centers[1])
            team = "A" if d0 <= d1 else "B"
            self._team_of_track[track_id] = team
            return team
        # sin información suficiente: no asignar (el pipeline lo ignora este frame)
        return None
