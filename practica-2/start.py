#iniciar el servidor con el comando uvicorn start:app --port 8125 --reload



import os

if __name__ == "__main__":
    # ruta donde esta este archivo ejecutandoce actualmente
    path = os.path.dirname(os.path.abspath(__file__))
    # iniciar el servidor ruta del archivo donde esta la clase FastAPI
    os.system(f"uvicorn main:app --port 8125 --app-dir {path}")
