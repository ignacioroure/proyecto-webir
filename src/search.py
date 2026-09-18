"""Busqueda de peliculas por similitud coseno contra los resumenes.

    python -m src.search "terror, ciudad"
    python -m src.search "terror, ciudad" --method both --top-k 5
    python -m src.search                      # modo interactivo

La idea central: la consulta se vectoriza con EXACTAMENTE el mismo
vectorizador que se uso para indexar, de modo que ambos vectores viven en el
mismo espacio y el coseno entre ellos es comparable.
"""

import argparse
import pickle
import sys
import textwrap

import numpy as np

from src.config import EMBEDDINGS_INDEX, TFIDF_INDEX
from src.corpus import load_movies


class TfidfSearcher:
    """Modelo de espacio vectorial clasico: bolsa de palabras ponderada."""

    name = "TF-IDF"

    def __init__(self) -> None:
        if not TFIDF_INDEX.exists():
            sys.exit("Falta el indice TF-IDF. Corré: python -m src.build_index")
        with TFIDF_INDEX.open("rb") as handle:
            payload = pickle.load(handle)
        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]

    def score(self, query: str) -> np.ndarray:
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            print(
                "  [aviso] ningun termino de la consulta esta en el vocabulario: "
                "TF-IDF no puede rankear nada.",
                file=sys.stderr,
            )
            return np.zeros(self.matrix.shape[0])
        # Ambas matrices estan L2-normalizadas -> producto punto == coseno.
        return (self.matrix @ query_vector.T).toarray().ravel()


class EmbeddingSearcher:
    """Busqueda semantica: vectores densos de un modelo pre-entrenado."""

    name = "Embeddings"

    def __init__(self) -> None:
        if not EMBEDDINGS_INDEX.exists():
            sys.exit("Falta el indice de embeddings. Corré: python -m src.build_index")
        payload = np.load(EMBEDDINGS_INDEX, allow_pickle=False)
        self.matrix = payload["matrix"]
        self.model_name = str(payload["model"])
        self._model = None

    @property
    def model(self):
        # Carga perezosa: importar torch tarda, y no hace falta si el usuario
        # solo pidio TF-IDF.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, query: str) -> np.ndarray:
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        return self.matrix @ query_vector


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices de los k mayores puntajes, ordenados de mayor a menor."""
    k = min(k, len(scores))
    # argpartition es O(n) y evita ordenar el corpus entero para quedarnos con k.
    candidates = np.argpartition(-scores, k - 1)[:k]
    ranked = candidates[np.argsort(-scores[candidates])]
    return [(int(i), float(scores[i])) for i in ranked]


def print_results(title: str, results: list[tuple[int, float]], movies: list[dict]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for rank, (index, score) in enumerate(results, start=1):
        movie = movies[index]
        year = movie.get("year") or "s/f"
        genres = ", ".join(movie.get("genres", [])) or "sin generos"
        print(f"{rank}. [{score:.4f}] {movie['title']} ({year})")
        print(f"   {genres}")
        summary = textwrap.shorten(movie["overview"], width=150, placeholder="...")
        print(f"   {textwrap.fill(summary, width=76, subsequent_indent='   ')}")


def run_query(query: str, method: str, k: int, searchers: dict, movies: list[dict]) -> None:
    print(f"\n{'=' * 78}\nConsulta: {query!r}\n{'=' * 78}")
    for key in ("tfidf", "embeddings"):
        if method in (key, "both"):
            searcher = searchers[key]
            print_results(searcher.name, top_k(searcher.score(query), k), movies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca peliculas por su resumen.")
    parser.add_argument("query", nargs="?", help="texto de la consulta")
    parser.add_argument(
        "--method", choices=["tfidf", "embeddings", "both"], default="both"
    )
    parser.add_argument("--top-k", type=int, default=5, help="resultados por metodo")
    args = parser.parse_args()

    movies = load_movies()
    searchers = {}
    if args.method in ("tfidf", "both"):
        searchers["tfidf"] = TfidfSearcher()
    if args.method in ("embeddings", "both"):
        searchers["embeddings"] = EmbeddingSearcher()

    if args.query:
        run_query(args.query, args.method, args.top_k, searchers, movies)
        return

    print("Modo interactivo. Escribí una consulta (Ctrl-C para salir).")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query:
            run_query(query, args.method, args.top_k, searchers, movies)


if __name__ == "__main__":
    main()
