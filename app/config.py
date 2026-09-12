"""Charge la configuration de l'application depuis config.yaml."""

import yaml

try:
    with open("config.yaml", "r") as file:
        config: dict = yaml.safe_load(file)

except FileNotFoundError:
    print("Le fichier config.yaml n'existe pas. Valeurs par défaut : aucune.")
    config = {}

