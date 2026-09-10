"""Cuantización geodésica "a posteriori" sobre los latentes del VAE.

La versión original tenía las funciones correctas para esto
(`kneighbors_graph` + `shortest_path` + MDS + K-means) en un archivo de
utilidades, pero el script de entrenamiento nunca las llamaba: en su lugar
ejecutaba K-means euclídeo normal sobre una submuestra de solo 1.000 puntos.
Es decir, el método "geodésico" que se comparaba en el informe nunca se
había ejecutado.

Aquí se conectan de verdad, con dos cambios para que sea viable
computacionalmente sobre un dataset de 60.000 imágenes:

1. **Landmark Isomap**: en vez de calcular la matriz de distancias
   geodésicas completa N×N (inviable en memoria para N=60.000: ocuparía
   ~14GB), se calculan las distancias geodésicas solo desde un subconjunto
   de "landmarks" (p. ej. 500) a todos los puntos, usando Dijkstra sobre el
   grafo k-NN disperso. Es el truco estándar de Landmark MDS/Isomap
   (de Silva & Tenenbaum, 2002) para escalar Isomap a datasets grandes.
2. El K-means final se ejecuta sobre TODOS los puntos embebidos, no sobre
   una submuestra de 1.000 — el codebook se ajusta con el dataset completo.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


def landmark_geodesic_embedding(
    X: np.ndarray,
    n_neighbors: int = 10,
    n_landmarks: int = 500,
    n_components: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    """Embebe X (N, D) en un espacio de `n_components` dimensiones que aproxima
    las distancias geodésicas sobre el manifold, usando Landmark MDS.

    Devuelve un array (N, n_components).
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    n_landmarks = min(n_landmarks, n)

    print(f"🔹 Construyendo grafo k-NN (k={n_neighbors}) sobre {n} puntos...")
    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance", include_self=False)
    # Simetrizar: si A es vecino de B pero no al revés, nos quedamos con la conexión
    knn_graph = knn_graph.maximum(knn_graph.T)

    landmark_idx = rng.choice(n, size=n_landmarks, replace=False)

    print(f"🔹 Calculando distancias geodésicas desde {n_landmarks} landmarks (Dijkstra)...")
    # shortest_path con `indices` calcula solo las filas pedidas -> O(L · N log N), no O(N² log N)
    D_landmarks = shortest_path(
        csr_matrix(knn_graph), method="D", directed=False, indices=landmark_idx
    )  # (n_landmarks, N)

    # Puntos inalcanzables (grafo desconectado) -> distancia = máxima finita observada
    finite_max = np.nanmax(np.where(np.isfinite(D_landmarks), D_landmarks, np.nan))
    D_landmarks = np.where(np.isfinite(D_landmarks), D_landmarks, finite_max * 2)

    print("🔹 Embebiendo con Landmark MDS...")
    # Landmark MDS (de Silva & Tenenbaum, 2002): MDS clásico sobre el bloque
    # landmark-landmark, y el resto de puntos se proyectan por triangulación.
    D_ll = D_landmarks[:, landmark_idx]  # (L, L)
    D_ll_sq = D_ll ** 2
    J = np.eye(n_landmarks) - np.ones((n_landmarks, n_landmarks)) / n_landmarks
    B = -0.5 * J @ D_ll_sq @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvals = np.clip(eigvals[order], a_min=1e-12, a_max=None)
    eigvecs = eigvecs[:, order]
    L_pseudo = eigvecs * np.sqrt(eigvals)  # (n_landmarks, n_components) — coords de los landmarks

    # Proyección del resto de puntos (Landmark MDS "distance-based triangulation")
    mean_sq_d = (D_ll_sq).mean(axis=1)  # (L,)
    D_sq = D_landmarks ** 2  # (L, N)
    proj = -0.5 * (D_sq.T - mean_sq_d[None, :]) @ (eigvecs / np.sqrt(eigvals))  # (N, n_components)

    return proj.astype(np.float32)


def geodesic_kmeans(
    X: np.ndarray,
    n_clusters: int = 64,
    n_neighbors: int = 10,
    n_landmarks: int = 500,
    n_components: int = 10,
    random_state: int = 0,
):
    """Clustering geodésico real sobre TODO el dataset X (N, D).

    Devuelve (labels, centroids_in_original_space, embedding).
    Los centroides se devuelven en el espacio original (promediando los
    puntos de cada cluster), para poder decodificarlos directamente con
    el decoder del VAE.
    """
    embedding = landmark_geodesic_embedding(
        X, n_neighbors=n_neighbors, n_landmarks=n_landmarks,
        n_components=n_components, random_state=random_state,
    )

    print(f"🔹 Ejecutando K-means (K={n_clusters}) sobre el embedding geodésico...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embedding)

    # Centroide en el espacio ORIGINAL (media de los puntos del cluster) para poder
    # decodificar con el VAE — el centroide del embedding MDS no es decodificable.
    rng = np.random.RandomState(random_state)
    centroids = np.zeros((n_clusters, X.shape[1]), dtype=np.float32)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            centroids[k] = X[mask].mean(axis=0)
        else:
            # Cluster vacío (raro, pero posible con K grande): usamos un punto
            # aleatorio del dataset en vez de dejarlo en cero.
            centroids[k] = X[rng.randint(0, X.shape[0])]
    return labels, centroids, embedding


def euclidean_kmeans(X: np.ndarray, n_clusters: int = 64, random_state: int = 0):
    """K-means euclídeo estándar sobre los mismos datos — se usa como control
    para el ablation geodésico vs. euclídeo (ver evaluation/evaluate.py)."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_.astype(np.float32)
