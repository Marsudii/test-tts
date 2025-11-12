import config
from fastapi import FastAPI
from app.pkg.tts_pkg import TtsLibrary

# from app.pkg import TtsLibrary

# FAST API Configuration
fastapp = FastAPI(
    title="Fast API Code Standard",
    description="TEXT TO SPEECH INDONESIA API",
    version="0.0.0",
    docs_url="/docs",
)


# # PKG Initialization
tts_lib = TtsLibrary(config)

# IMPORT ROUTES
from .controllers import tts_controller


#  ADD ROUTES
fastapp.include_router(tts_controller.router)
