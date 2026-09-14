"""Charge la configuration de l'application depuis config.yaml."""

from pathlib import Path

import yaml

# Chemin absolu vers config.yaml (racine du projet), pour que le
# chargement marche peu importe le répertoire courant depuis lequel le
# script est lancé (ex: `python main.py` depuis app/, ou pytest depuis
# la racine).
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

try:
    with open(CONFIG_PATH, "r") as file:
        config: dict = yaml.safe_load(file)

except FileNotFoundError:
    print(f"Le fichier config.yaml n'existe pas ({CONFIG_PATH}). Valeurs par défaut : aucune.")
    config = {}

