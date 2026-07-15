import os
import httpx

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SPLATS_BUCKET = os.environ.get("SPLATS_BUCKET", "splats")


async def upload_splat_file(file_path: str, model_id: str) -> str:
    storage_path = f"{model_id}.ply"

    with open(file_path, "rb") as f:
        content = f.read()

    async with httpx.AsyncClient(verify=False, timeout=60) as client:
        response = await client.post(
            f"{SUPABASE_URL}/storage/v1/object/{SPLATS_BUCKET}/{storage_path}",
            content=content,
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/octet-stream",
            },
        )
        response.raise_for_status()

    return storage_path