import os
import asyncio
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm
from .utils import CooldownLimiter, worker
from google.genai import Client
from google.genai.types import HttpOptions, HttpRetryOptions
from dotenv import load_dotenv
import json

def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def get_logs() -> list[dict]:
    log = {
        'task': 'overtake',
        'ego_idx': 8,
        'target_idx': 0,
        'command': 'Overtake the car in front of ego vehicle',
        'video_path': '/zfsauton2/home/eshau/gemini/example_videos/training_tfexample.tfrecord-00100-of-01000.scenario_025.target_000.mp4',
        'tfrecord': 'training_tfexample.tfrecord-00100-of-01000',
        'scenario_idx': 25
    }
    return [log, log]


async def main() -> None:
    # Change to RPM from Google for whatever model we are using:
    # https://aistudio.google.com/u/0/rate-limit?timeRange=last-28-days
    # Default is "gemini-3.1-flash-lite-preview"
    rpm_limit = 15
    limiter = CooldownLimiter(min_interval=60.0 / rpm_limit)

    # change to idk
    num_concurrent_workers = 3

    logs = get_logs()   # change to how we're actually getting logs

    http_options = HttpOptions(
        retry_options= HttpRetryOptions(
            attempts=6,              # total attempts including first try
            initial_delay=2.0,       # seconds
            max_delay=30.0,          # seconds
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=[429, 503],
        )
    )
    queue = asyncio.Queue()
    results = [{} for _ in logs]

    for i, log in enumerate(logs):
        await queue.put((i, log))

    async with Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=http_options
    ).aio as client:
        with tqdm(total=len(logs), desc='Processing logs') as pbar:
            workers = [
                asyncio.create_task(
                    worker(
                        f'worker-{i+1}',
                        client, queue, limiter, results, pbar
                ))
                for i in range(num_concurrent_workers)  # max concurrent workers
            ]

        for _ in workers:
            await queue.put(None)

        await queue.join()
        await asyncio.gather(*workers)

    for i, result in enumerate(results):
        print(f"\n=== Log {i} ===")
        print(result)

async def query_gemini(log_paths):
    rpm_limit = 15
    limiter = CooldownLimiter(min_interval=60.0 / rpm_limit)

    # change to idk
    num_concurrent_workers = 3

    logs = [json.load(open(log_path)) for log_path in log_paths]

    http_options = HttpOptions(
        retry_options= HttpRetryOptions(
            attempts=6,              # total attempts including first try
            initial_delay=2.0,       # seconds
            max_delay=30.0,          # seconds
            exp_base=2.0,
            jitter=1.0,
            http_status_codes=[429, 503],
        )
    )
    queue = asyncio.Queue()
    results = [{} for _ in logs]

    for i, log in enumerate(logs):
        await queue.put((i, log))

    async with Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=http_options
    ).aio as client:
        workers = [
            asyncio.create_task(
                worker(
                    f'worker-{i+1}',
                    client, queue, limiter, results
            ))
            for i in range(num_concurrent_workers)  # max concurrent workers
        ]

        for _ in workers:
            await queue.put(None)

        await queue.join()
        await asyncio.gather(*workers)

    for i, result in enumerate(results):
        _save_json(Path(log_paths[i]).with_suffix('.result.json'), result)
        


if __name__ == '__main__':
    load_dotenv()
    asyncio.run(main())