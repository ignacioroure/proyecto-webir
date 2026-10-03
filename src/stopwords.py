"""Lista de palabras vacias en espanol para el vectorizador TF-IDF.

scikit-learn solo trae stopwords en ingles, asi que definimos las nuestras.
Sin esto, terminos como "de", "la" o "que" dominan el vocabulario y ensucian
la similitud entre documentos.

La base es la lista de Snowball, el estandar para espanol. Se le suman
palabras que en este corpus aparecen en decenas de resumenes sin aportar
significado ("despues", "puede", "hace"...), encontradas ordenando el
vocabulario por frecuencia de documento.
"""

from sklearn.feature_extraction.text import strip_accents_unicode

# https://snowballstem.org/algorithms/spanish/stop.txt
_SNOWBALL = """
de la que el en y a los del se las por un para con no una su al lo como más
pero sus le ya o este sí porque esta entre cuando muy sin sobre también me
hasta hay donde quien desde todo nos durante todos uno les ni contra otros
ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él
tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas
algo nosotros mi mis tú te ti tu tus ellas nosotras vosotros vosotras os mío
mía míos mías tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra
nuestros nuestras vuestro vuestra vuestros vuestras esos esas
estoy estás está estamos estáis están esté estés estemos estéis estén estaré
estarás estará estaremos estaréis estarán estaría estarías estaríamos
estaríais estarían estaba estabas estábamos estabais estaban estuve estuviste
estuvo estuvimos estuvisteis estuvieron estuviera estuvieras estuviéramos
estuvierais estuvieran estuviese estuvieses estuviésemos estuvieseis
estuviesen estando estado estada estados estadas estad
he has ha hemos habéis han haya hayas hayamos hayáis hayan habré habrás habrá
habremos habréis habrán habría habrías habríamos habríais habrían había
habías habíamos habíais habían hube hubiste hubo hubimos hubisteis hubieron
hubiera hubieras hubiéramos hubierais hubieran hubiese hubieses hubiésemos
hubieseis hubiesen habiendo habido habida habidos habidas
soy eres es somos sois son sea seas seamos seáis sean seré serás será seremos
seréis serán sería serías seríamos seríais serían era eras éramos erais eran
fui fuiste fue fuimos fuisteis fueron fuera fueras fuéramos fuerais fueran
fuese fueses fuésemos fueseis fuesen siendo sido
tengo tienes tiene tenemos tenéis tienen tenga tengas tengamos tengáis tengan
tendré tendrás tendrá tendremos tendréis tendrán tendría tendrías tendríamos
tendríais tendrían tenía tenías teníamos teníais tenían tuve tuviste tuvo
tuvimos tuvisteis tuvieron tuviera tuvieras tuviéramos tuvierais tuvieran
tuviese tuvieses tuviésemos tuvieseis tuviesen teniendo tenido tenida tenidos
tenidas tened
"""

# La primera linea son palabras de la lista original que Snowball no trae; el
# resto, palabras frecuentes en los resumenes de este corpus sin contenido
# propio. "the" viene de los titulos en ingles.
_EXTRA = """
aunque cada dos mientras según ser solo tras
después ahora pronto siempre nunca entonces allí allá aquí acá así aún tan
luego además casi incluso hacia mediante junto embargo vez
toda todas unas mismo misma mismos mismas ambos ambas cualquier cualquiera
cuyo cuya cuyos cuyas demás
hace hacen hacer puede pueden podrá podría debe deben deberá deberán va van ve
película películas the
"""

# El vectorizador quita los acentos antes de filtrar las stopwords, asi que la
# lista tiene que estar igual de normalizada: si no, "más" nunca coincidiria
# con el token "mas".
SPANISH_STOPWORDS = sorted(
    {strip_accents_unicode(word) for word in (_SNOWBALL + _EXTRA).split()}
)
