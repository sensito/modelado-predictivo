#iniciar el servidor con el comando uvicorn start:app --port 8125 --reload

import os

if __name__ == "__main__":
    os.system("uvicorn main:app --port 8125 --reload") 