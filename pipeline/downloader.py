import os
import httpx


async def download_video(video_url: str, work_dir: str) -> str:
    video_path = os.path.join(work_dir, "source_video.mp4")

    # verify=False porque el backend .NET usa un certificado autofirmado
    # en desarrollo local -- en producción, con un certificado real, quita esto.
    async with httpx.AsyncClient(verify=False, timeout=120) as client:
        async with client.stream("GET", video_url) as response:
            response.raise_for_status()
            with open(video_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

    return video_path