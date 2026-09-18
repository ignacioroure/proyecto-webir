# Buscador semántico de películas

Trabajo práctico de **Recuperación de Información y Recomendaciones en la Web**.

Un buscador que recibe una consulta en lenguaje natural (`"terror, ciudad"`) y
devuelve las películas cuyos **resúmenes** son más parecidos, ordenadas por
similitud coseno. Implementa y compara dos formas de representar el texto como
vectores:

| | TF-IDF | Embeddings |
|---|---|---|
| Representación | Dispersa, ~miles de dimensiones | Densa, 384 dimensiones |
| Qué captura | Coincidencia **léxica** (palabras compartidas) | Coincidencia **semántica** (significado) |
| Vocabulario | Aprendido del corpus | Pre-entrenado, multilingüe |
| Interpretable | Sí: se puede ver qué término aportó | No directamente |
| Costo | Instantáneo | Requiere descargar un modelo (~450 MB) |

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

El repo incluye un corpus de demostración de 12 películas, así que se puede
probar sin API key:

```bash
python -m src.build_index
python -m src.search "terror, ciudad"
```

Para el corpus real desde TMDB:

```bash
cp .env.example .env     # y pegar la API key adentro
python -m src.fetch_movies --target 800
python -m src.build_index
python -m src.search "terror, ciudad" --top-k 10
```

Sin argumentos, `src.search` entra en modo interactivo. Con `--method tfidf` o
`--method embeddings` corre uno solo de los dos.

Para conseguir la API key: registrarse en [TMDB](https://www.themoviedb.org/signup)
y generarla en [Configuración → API](https://www.themoviedb.org/settings/api).
Es gratuita.

## Arquitectura

```
src/fetch_movies.py   Descarga N películas de TMDB      -> data/movies.json
src/corpus.py         Carga el corpus y arma el texto indexable
src/build_index.py    Vectoriza los resúmenes           -> index/*.pkl, *.npz
src/search.py         Vectoriza la consulta y rankea por coseno
```

El punto clave del diseño: **la consulta se vectoriza con exactamente el mismo
vectorizador que se usó para indexar**. Solo así ambos vectores viven en el
mismo espacio y el coseno entre ellos significa algo.

Como todos los vectores se guardan normalizados a norma 1, la similitud coseno
se reduce a un producto punto, y el ranking completo del corpus es una sola
multiplicación matriz-vector.

## Decisiones de diseño

- **Qué se indexa.** No solo el resumen: se concatena `título + géneros +
  resumen`. Los géneros son una señal corta pero muy informativa, y hacen que
  una consulta como `"terror"` matchee aunque el resumen nunca use la palabra.
- **Stopwords propias.** scikit-learn solo trae lista en inglés
  (`src/stopwords.py`). Sin ella, términos como *de*, *la* o *que* dominan el
  vocabulario.
- **`sublinear_tf=True`.** Pondera `1 + log(tf)` en vez de la frecuencia cruda:
  que un término aparezca 10 veces no lo hace 10 veces más relevante.
- **`min_df=2`, `max_df=0.5`.** Descartan términos que aparecen en un solo
  documento (ruido, nombres propios irrepetibles) y los demasiado frecuentes.
- **Modelo multilingüe.** `paraphrase-multilingual-MiniLM-L12-v2` mapea español
  e inglés al mismo espacio, así que la consulta puede estar en un idioma y el
  resumen en otro.

## Resultado de ejemplo

Consulta `"terror, ciudad"` sobre el corpus de demostración:

| # | TF-IDF | Embeddings |
|---|---|---|
| 1 | Sombras en el subte (0.530) | La hora del apagón (0.545) |
| 2 | **Casa vacía (0.454)** | Sombras en el subte (0.449) |
| 3 | La hora del apagón (0.382) | Robo a mano armada (0.436) |

*Casa vacía* es terror **rural**: TF-IDF la rankea 2ª porque matcheó el término
"terror" e ignoró por completo "ciudad", que no aparece literalmente en ningún
resumen. Los embeddings ponen 1ª *La hora del apagón*, la única que cumple las
dos condiciones a la vez — y lo hace sin compartir una sola palabra con la
consulta: su resumen dice "megalópolis", no "ciudad".

El caso inverso también aparece: los embeddings cuelan *Robo a mano armada*
(crimen, no terror) en el top-3, porque "asalto" y "banda" quedan cerca de
"terror" en el espacio semántico. Es el costo de la generalización.

## Limitaciones conocidas

- El corpus de demostración es demasiado chico para que TF-IDF luzca: con 12
  documentos y `min_df=2` el vocabulario queda en ~14 términos. Con 800
  películas pasa a varios miles.
- No hay evaluación cuantitativa (precisión@k, MAP) porque no existe un
  conjunto de relevancia etiquetado. Construir uno a mano sobre unas pocas
  consultas sería el siguiente paso natural.
- La búsqueda es exhaustiva: se compara contra todo el corpus. Con este tamaño
  sobra; a partir de ~10⁵ documentos haría falta un índice aproximado (FAISS,
  HNSW).
