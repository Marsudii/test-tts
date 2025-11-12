# APP/CONTROLLERS/TTS_CONTROLLER.PY
import config
from fastapi import APIRouter, Response
from app import tts_lib

# API ROUTER INITIALIZATION
router = APIRouter(
    prefix="/api",
    tags=["TTS INDONESIA"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal Server Error"},
    },
)


@router.post("/generate")
def generate_tts(body: dict):
    text = body.get("text", "")

    if not text:
        return {"error": "text is required"}

    buf, sr = tts_lib.generate_audio(text)
    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )
