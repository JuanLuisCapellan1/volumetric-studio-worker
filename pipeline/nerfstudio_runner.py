import os
import subprocess

NERFSTUDIO_IMAGE = os.environ.get("NERFSTUDIO_IMAGE", "ghcr.io/nerfstudio-project/nerfstudio:latest")
MAX_TRAIN_ITERATIONS = os.environ.get("MAX_TRAIN_ITERATIONS", "15000")
MIN_REGISTERED_FRAMES = 20

def _run_docker_command(work_dir: str, args: list[str], timeout_seconds: int) -> None:
    volume_mount = f"{work_dir}:/workspace/"

    docker_args = [
        "docker", "run", "--rm", "--gpus", "all",
        "--shm-size=12gb",
        "-v", volume_mount,
        NERFSTUDIO_IMAGE,
    ] + args

    result = subprocess.run(
        docker_args,
        capture_output=True,
        text=True,
        encoding="utf-8",   # ← fuerza UTF-8 en vez de cp1252 (default en Windows)
        errors="replace",   # ← sustituye caracteres no decodificables en vez de fallar
        timeout=timeout_seconds,
    )

    # Blindaje extra: nunca asumir que stdout/stderr no son None
    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""

    print(f"--- STDOUT ({args[0]}) ---\n{stdout_text[-3000:]}")
    print(f"--- STDERR ({args[0]}) ---\n{stderr_text[-3000:]}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Comando de nerfstudio falló (exit code {result.returncode}):\n"
            f"STDOUT (final): {stdout_text[-2000:]}\n"
            f"STDERR (final): {stderr_text[-2000:]}"
        )

def process_camera_poses(work_dir: str) -> None:
    """
    Ejecuta ns-process-data sobre el video fuente. Esto REEMPLAZA nuestra
    extracción de frames con ffmpeg: nerfstudio hace su propia selección de
    frames (evitando borrosos/duplicados) y corre COLMAP internamente para
    resolver la posición de cada cámara (Structure-from-Motion).
    """
    _run_docker_command(
        work_dir,
        ["ns-process-data", "video", "--data", "/workspace/source_video.mp4", "--output-dir", "/workspace/processed"],
        timeout_seconds=900,
    )

def validate_camera_poses(work_dir: str) -> None:
    transforms_path = os.path.join(work_dir, "processed", "transforms.json")

    if not os.path.exists(transforms_path):
        raise RuntimeError("COLMAP no generó transforms.json -- fallo total de alineación de cámaras")

    with open(transforms_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_count = len(data.get("frames", []))
    if frame_count < MIN_REGISTERED_FRAMES:
        raise RuntimeError(
            f"COLMAP solo pudo registrar {frame_count} imágenes de cámara -- "
            f"insuficiente para entrenar (mínimo recomendado: {MIN_REGISTERED_FRAMES}). "
            "Esto suele deberse a poca cobertura de la escena, movimiento demasiado "
            "rápido, poca superposición entre frames, o iluminación inconsistente."
        )

def train_splat_model(work_dir: str) -> str:
    """
    Entrena con Splatfacto (implementación de 3D Gaussian Splatting de
    nerfstudio). --max-num-iterations en 15000 es un punto de partida
    conservador para 8GB VRAM (RTX 3070) -- ajusta según tiempo/calidad
    que necesites.
    """
    _run_docker_command(
        work_dir,
        [
            "ns-train", "splatfacto",
            "--data", "/workspace/processed",
            "--output-dir", "/workspace/output",
            "--max-num-iterations", MAX_TRAIN_ITERATIONS,
            "--viewer.quit-on-train-completion", "True",
            "--vis", "tensorboard",
        ],
        timeout_seconds=3600,
    )

    return _find_latest_config(work_dir)


def export_splat_file(work_dir: str, config_path_in_container: str) -> str:
    """Exporta el modelo entrenado a un .ply consumible por el visor web."""
    _run_docker_command(
        work_dir,
        ["ns-export", "gaussian-splat", "--load-config", config_path_in_container, "--output-dir", "/workspace/export"],
        timeout_seconds=600,
    )

    export_file = os.path.join(work_dir, "export", "splat.ply")
    if not os.path.exists(export_file):
        raise FileNotFoundError("ns-export no generó el archivo splat.ply esperado")

    return export_file


def _find_latest_config(work_dir: str) -> str:
    output_root = os.path.join(work_dir, "output", "processed", "splatfacto")
    if not os.path.isdir(output_root):
        raise FileNotFoundError(f"No se encontró el directorio de salida de ns-train: {output_root}")

    timestamps = sorted(os.listdir(output_root))
    if not timestamps:
        raise FileNotFoundError("ns-train no generó ninguna carpeta de resultados")

    latest = timestamps[-1]
    host_config_path = os.path.join(output_root, latest, "config.yml")

    if not os.path.exists(host_config_path):
        raise FileNotFoundError(f"No se encontró config.yml en {host_config_path}")

    # Devolvemos la ruta EN EL CONTENEDOR (/workspace/...), que es lo que
    # ns-export necesita recibir en --load-config
    return f"/workspace/output/processed/splatfacto/{latest}/config.yml"