# config/logging.py
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt=f"%Y-%m-%d %H:%M:%S",
        force=True,  # asegura que se sobrescriba cualquier config previa
    )
