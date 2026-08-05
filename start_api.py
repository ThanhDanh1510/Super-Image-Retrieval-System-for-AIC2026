"""Start the API server — IN-PROCESS (1 process, 1 venv, tiết kiệm RAM)."""
import logging

import uvicorn

from serving.api.core.config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()

    print(f"Starting API server in-process: http://{settings.HOST}:{settings.PORT}")
    print("Models will lazy-load on first request (RAM saver).")
    print(f"Check health at http://localhost:{settings.PORT}/health")

    # uvicorn.run với import string + workers=1 + không reload => chạy NGAY
    # trong tiến trình hiện tại, KHÔNG spawn thêm process Python nào.
    uvicorn.run(
        "serving.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
    )


if __name__ == "__main__":
    main()
