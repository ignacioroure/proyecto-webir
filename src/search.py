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


def check_corpus(index_ids, movies: list[dict], index_name: str) -> None:
    """Corta si el indice no se construyo sobre el corpus actual.

    La fila i del indice es la pelicula movies[i]. Si el corpus cambia (por
    ejemplo, al volver a correr fetch_movies) y no se reindexa, el buscador
    mostraria peliculas equivocadas sin ningun error.
    """
    if index_ids is None or not np.array_equal(index_ids, [m["id"] for m in movies]):
        sys.exit(
            f"El indice {index_name} no corresponde al corpus actual (o es de una "
            "version anterior). Corré: python -m src.build_index"
        )


def no_candidates(allowed: np.ndarray | None) -> bool:
    """True si el filtro no dejo ninguna pelicula en carrera.

    Puntuar igual seria trabajo tirado, y en el caso de los embeddings cuesta
    un forward pass del modelo (cientos de milisegundos) para devolver [].
    """
    return allowed is not None and not allowed.any()


class TfidfSearcher:
    """Modelo de espacio vectorial clasico: bolsa de palabras ponderada."""

    name = "TF-IDF"

    def __init__(self, movies: list[dict]) -> None:
        if not TFIDF_INDEX.exists():
            sys.exit("Falta el indice TF-IDF. Corré: python -m src.build_index")
        with TFIDF_INDEX.open("rb") as handle:
            payload = pickle.load(handle)
        check_corpus(payload.get("ids"), movies, self.name)
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

    def search(
        self, query: str, k: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        if no_candidates(allowed):
            return []
        # Puntaje 0 = ningun termino en comun con la consulta: esas peliculas
        # serian relleno, no resultados.
        results = top_k(self.score(query), k, allowed)
        return [(index, score) for index, score in results if score > 0]


class EmbeddingSearcher:
    """Busqueda semantica: vectores densos de un modelo pre-entrenado."""

    name = "Embeddings"

    def __init__(self, movies: list[dict]) -> None:
        if not EMBEDDINGS_INDEX.exists():
            sys.exit("Falta el indice de embeddings. Corré: python -m src.build_index")
        payload = np.load(EMBEDDINGS_INDEX, allow_pickle=False)
        check_corpus(payload.get("ids"), movies, self.name)
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

    def search(
        self, query: str, k: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        if no_candidates(allowed):
            return []
        return top_k(self.score(query), k, allowed)


def top_k(
    scores: np.ndarray, k: int, allowed: np.ndarray | None = None
) -> list[tuple[int, float]]:
    """Indices de los k mayores puntajes, ordenados de mayor a menor.

    allowed es una mascara booleana opcional: solo compiten las peliculas en
    True. Filtrar antes de elegir los k (y no despues) evita quedarse con 1 de 5.
    """
    candidates = np.arange(len(scores)) if allowed is None else np.flatnonzero(allowed)
    k = min(k, len(candidates))
    if k < 1:
        return []
    subset = scores[candidates]
    # argpartition es O(n) y evita ordenar el corpus entero para quedarnos con k.
    best = np.argpartition(-subset, k - 1)[:k]
    ranked = best[np.argsort(-subset[best])]
    return [(int(candidates[i]), float(subset[i])) for i in ranked]


def print_results(title: str, results: list[tuple[int, float]], movies: list[dict]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not results:
        print("(sin resultados)")
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
            print_results(searcher.name, searcher.search(query, k), movies)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("tiene que ser un entero mayor o igual a 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca peliculas por su resumen.")
    parser.add_argument("query", nargs="?", help="texto de la consulta")
    parser.add_argument(
        "--method", choices=["tfidf", "embeddings", "both"], default="both"
    )
    parser.add_argument(
        "--top-k", type=positive_int, default=5, help="resultados por metodo"
    )
    args = parser.parse_args()

    movies = load_movies()
    searchers = {}
    if args.method in ("tfidf", "both"):
        searchers["tfidf"] = TfidfSearcher(movies)
    if args.method in ("embeddings", "both"):
        searchers["embeddings"] = EmbeddingSearcher(movies)

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
