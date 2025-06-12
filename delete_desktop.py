# delete_desktop_ini.py
import pathlib, os
root = pathlib.Path("mlruns")
for p in root.rglob("desktop.ini"):
    try:
        os.remove(p)
        print("Eliminado:", p)
    except OSError as e:
        print("No se pudo borrar", p, "→", e)
