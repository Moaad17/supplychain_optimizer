import yaml

try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

except FileNotFoundError:
    print("Le fichier config.yaml n'existe pas.")

