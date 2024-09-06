#Recherche semantique avec pour source de donnée les urls des jeux de la base de donnée gamewisher 

- Connection à la bdd Gamewisher avec mongoose.
- Chargement de 100 jeux (trop long avec la totalité, ~12k)
- Transforme les jeux avec un texte contentant le tittre et la sescription avec le converter tensor flow de base.
- Transforme la requete écrite par l'utilisateur dans le terminal 
- Effectue une recherche par similitude avec une approche cosine.
- Affiche les 3 résulats avec le score le plus haut.

Necessaire : fichier .env 

MONGODB_URI=<mongodb uri>
// 100 par defaut dans le code, mais configurable.
// Au dela, le traitement initial est extremement long
LIMIT=<limit> 
