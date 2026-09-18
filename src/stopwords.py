"""Lista de palabras vacias en espanol para el vectorizador TF-IDF.

scikit-learn solo trae stopwords en ingles, asi que definimos las nuestras.
Sin esto, terminos como "de", "la" o "que" dominan el vocabulario y ensucian
la similitud entre documentos.
"""

SPANISH_STOPWORDS = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aunque", "cada",
    "como", "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
    "dos", "el", "ella", "ellas", "ellos", "en", "entre", "era", "eran", "es",
    "esa", "esas", "ese", "eso", "esos", "esta", "estan", "estas", "este",
    "esto", "estos", "fue", "fueron", "ha", "han", "hasta", "hay", "la", "las",
    "le", "les", "lo", "los", "mas", "me", "mi", "mientras", "muy", "nada",
    "ni", "no", "nos", "nuestra", "nuestro", "o", "otra", "otro", "para",
    "pero", "poco", "por", "porque", "que", "quien", "se", "segun", "ser",
    "si", "sin", "sobre", "solo", "son", "su", "sus", "tambien", "tanto", "te",
    "tiene", "tienen", "todo", "todos", "tras", "tu", "un", "una", "uno",
    "unos", "y", "ya", "yo",
]
