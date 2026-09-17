from pathlib import Path

import pandas as pd


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Charge un fichier CSV ou Excel et convertit sa colonne 'date' au
    format datetime.

    Parameters
    ----------
    filepath : str ou Path
        Chemin du fichier à charger. Formats acceptés : .csv, .xlsx, .xls.

    Returns
    -------
    pd.DataFrame
        Les données chargées, avec 'date' en datetime.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le format n'est pas supporté, ou si la colonne 'date' est
        absente du fichier.
    """

    filepath = Path(filepath)

    # Vérifier que le fichier existe
    if not filepath.exists():
        raise FileNotFoundError(
            f"Le fichier n'existe pas : {filepath}"
        )

    # Détecter l'extension
    extension = filepath.suffix.lower()

    # Charger le fichier
    if extension == ".csv":
        df = pd.read_csv(filepath)

    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)

    else:
        raise ValueError(
            f"Format non supporté : {extension}. "
            "Formats acceptés : .csv, .xlsx, .xls"
        )

    # Convertir la colonne date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    else:
        raise ValueError(
            "La colonne 'date' est absente du fichier."
        )

    return df