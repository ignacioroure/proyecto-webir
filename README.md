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

## Resultados

Medidos sobre el corpus real: 800 películas de TMDB, resúmenes en español,
vocabulario TF-IDF de 5407 términos.

### Consulta por keywords: `"terror, ciudad"`

| # | TF-IDF | Embeddings |
|---|---|---|
| 1 | Scream (0.222) | Insidious: Fuera del más allá (0.624) |
| 2 | Resident Evil: Raccoon City (0.194) | Cuenta atrás (0.599) |
| 3 | Weapons (0.187) | Terrifier 3 (0.577) |

Gana TF-IDF: sus tres primeros cumplen las dos condiciones a la vez. Los
embeddings traen buen terror pero fallan en lo urbano — *Hokum*, 4ª, transcurre
en una posada rural irlandesa.

### Consulta descriptiva: `"un padre que hace lo imposible por recuperar a su hija"`

| # | TF-IDF | Embeddings |
|---|---|---|
| 1 | **Venganza (0.252)** | Señora Doubtfire (0.630) |
| 2 | Kraven the Hunter (0.132) | **Venganza (0.566)** |
| 3 | Aladdin (0.126) | A la carrera (0.545) |

Ambos encuentran *Venganza*, que es la respuesta correcta. TF-IDF la pone 1ª con
el doble de puntaje que la siguiente; los embeddings la ponen 2ª.

### Lectura

Ninguno de los dos domina. Lo que cambia es **cómo fallan**:

- **TF-IDF matchea términos aislados y pierde la relación entre ellos.**
  *Aladdín* entra en el top-3 porque su resumen contiene "hija" y habla de un
  padre (el sultán), aunque la trama no tenga nada que ver.
- **Los embeddings capturan el tema general y pierden el específico.**
  *Señora Doubtfire* encabeza porque es intensamente "sobre un padre", pero no
  hay ninguna hija que recuperar.

Un detalle metodológico que conviene registrar: la ventaja de los embeddings
era mucho más clara sobre el corpus de demostración de 12 películas, y **no se
sostuvo al escalar a 800**. Con un corpus grande de películas populares hay
suficiente coincidencia léxica como para que TF-IDF resuelva bien las consultas
por keywords. Sacar conclusiones de un corpus chico habría sido un error.

### Nota sobre los puntajes

Los puntajes de las dos columnas **no son comparables entre sí**. Ambos son
cosenos, pero sobre espacios distintos: en TF-IDF dos documentos solo comparten
dirección si comparten términos, así que los valores se concentran cerca de 0;
los embeddings densos ocupan un cono estrecho del espacio y arrancan alto. Lo
comparable es el **orden** dentro de cada columna, no la magnitud entre ellas.

## Limitaciones conocidas

- El corpus de demostración (12 películas) solo sirve para verificar que el
  pipeline corre: con `min_df=2` el vocabulario queda en 14 términos. Cualquier
  conclusión sobre los métodos hay que sacarla del corpus completo.
- El corpus son las 800 películas más populares de TMDB, o sea un sesgo fuerte
  hacia cine comercial reciente y de Hollywood. Los resultados no se
  generalizan a un catálogo arbitrario.
- No hay evaluación cuantitativa (precisión@k, MAP) porque no existe un
  conjunto de relevancia etiquetado. Construir uno a mano sobre unas pocas
  consultas sería el siguiente paso natural.
- La búsqueda es exhaustiva: se compara contra todo el corpus. Con este tamaño
  sobra; a partir de ~10⁵ documentos haría falta un índice aproximado (FAISS,
  HNSW).
