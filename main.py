import asyncio
import os
import shutil
import tempfile

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

from pipeline.downloader import download_video
from pipeline.nerfstudio_runner import process_camera_poses, validate_camera_poses, train_splat_model, export_splat_file
from pipeline.uploader import upload_splat_file


DOTNET_API_BASE_URL = os.environ["DOTNET_API_BASE_URL"]
DOTNET_INTERNAL_API_KEY = os.environ["DOTNET_INTERNAL_API_KEY"]
WORKER_API_KEY = os.environ["WORKER_API_KEY"]

app = FastAPI(title="Volumetric Studio - AI Worker")


class ProcessRequest(BaseModel):
    modelId: str
    videoUrl: str


async def report_progress(
    model_id: str, stage: str, progress: int,
    splat_storage_path: str | None = None, error_message: str | None = None,
) -> None:
    url = f"{DOTNET_API_BASE_URL}/internal/models/{model_id}/progress"
    payload = {
        "stage": stage, "progress": progress,
        "splatStoragePath": splat_storage_path, "errorMessage": error_message,
    }
    headers = {"X-Internal-Api-Key": DOTNET_INTERNAL_API_KEY}

    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()


async def run_pipeline(model_id: str, video_url: str) -> None:
    # tempfile.gettempdir() en vez de "/tmp" -- en Windows resuelve a
    # algo como C:\Users\<user>\AppData\Local\Temp
    work_dir = os.path.join(tempfile.gettempdir(), "volumetric-studio", model_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        await report_progress(model_id, "ExtractingFrames", 10)
        video_path = await download_video(video_url, work_dir)

        # nerfstudio espera este nombre exacto dentro de /workspace
        final_video_path = os.path.join(work_dir, "source_video.mp4")
        if video_path != final_video_path:
            os.rename(video_path, final_video_path)

        await report_progress(model_id, "AligningCameras", 30)
        await asyncio.to_thread(process_camera_poses, work_dir)
        await asyncio.to_thread(validate_camera_poses, work_dir)
        
        await report_progress(model_id, "TrainingModel", 60)
        config_path_in_container = await asyncio.to_thread(train_splat_model, work_dir)

        await report_progress(model_id, "GeneratingOutput", 90)
        splat_file_path = await asyncio.to_thread(export_splat_file, work_dir, config_path_in_container)

        storage_path = await upload_splat_file(splat_file_path, model_id)

        await report_progress(model_id, "Completed", 100, splat_storage_path=storage_path)

    except Exception as ex:
        await report_progress(model_id, "Failed", 0, error_message=str(ex))
        raise
    finally:
        pass
        #shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/process", status_code=202)
async def process_video(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    x_worker_api_key: str = Header(default=None, alias="X-Worker-Api-Key"),
):
    if x_worker_api_key != WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")

    background_tasks.add_task(run_pipeline, request.modelId, request.videoUrl)
    return {"message": "Procesamiento iniciado", "modelId": request.modelId}


@app.get("/health")
async def health_check():
    return {"status": "ok"}