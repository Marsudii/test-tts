import uvicorn
from uvicorn.config import LOGGING_CONFIG
from app import fastapp as app
from config import APP_PORT, APP_HOST


def run():
    try:
        uvicorn.run(
            app,
            log_config=LOGGING_CONFIG,
            host=APP_HOST,
            port=APP_PORT,
        )
    except KeyboardInterrupt:
        print("Server is shutting down...")


if __name__ == "__main__":
    run()
