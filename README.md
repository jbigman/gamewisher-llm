#Recherche semantique avec pour source de donnée les urls des jeux de la base de donnée gamewisher 

- Connection à la bdd Gamewisher avec mongoose.
- Chargement de 100 jeux (trop long avec la totalité, ~12k)
- Transforme les 100 titres avec le converter tensor flow de base.
- Transforme la requete faite par l'utilisateur en prompt 
- Effectue une recherche par similitude avec une approche cosine.
- Affiche les 3 résulats avec le score le plus haut.

