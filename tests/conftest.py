import sys
from pathlib import Path

# Le code de l'app utilise des imports "à plat" (from data.loader import ...,
# from forecasting.selector import ...) en supposant que app/ est à la
# racine du sys.path, comme quand on lance `python main.py` depuis app/.
# On reproduit ça ici pour que les tests importent les modules pareil.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))
