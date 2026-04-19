import uvicorn
import multiprocessing
from app.main import app

if __name__ == '__main__':
    multiprocessing.freeze_support()
    uvicorn.run(app, host="127.0.0.1", port=8000)
