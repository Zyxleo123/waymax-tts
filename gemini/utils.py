import os
import json
import time
import asyncio
from pathlib import Path
from enum import StrEnum
from tqdm.auto import tqdm
from operator import itemgetter
from pydantic import BaseModel, Field
from google.genai.client import AsyncClient
from google.genai.types import (
    Part, GenerateContentConfig,
    Blob, VideoMetadata, ThinkingConfig,
    MediaResolution
)


class Task(StrEnum):
    OVERTAKE = 'overtake'
    GIVE_WAY = 'give_way'
    PULL_OVER = 'pull_over'
    FOLLOW_LANE = 'follow_lane'
    CHANGE_LANE_LEFT = 'change_lane_left'
    CHANGE_LANE_RIGHT = 'change_lane_right'

class ActorCode(BaseModel):
    reason: str = Field(
        description='The reasoning behind the given code.'
    )
    code: str = Field(
        description='The new reward function to safely accomplish the task, given the command.'
    )

actor_schema = ActorCode.model_json_schema()

class CriticAnalysis(BaseModel):
    reason: str = Field(
        description='The reasoning behind the given command.'
    )
    command: str = Field(
        description='The new command given to safely accomplish the task.'
    )

critic_schema = CriticAnalysis.model_json_schema()

class CooldownLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
            self._next_time = max(now, self._next_time) + self.min_interval


MODULE_DIR = Path(__file__).resolve().parent


def _resolve_sample_dir(sample_path: str) -> Path:
    path = Path(sample_path)
    if not path.is_absolute():
        path = MODULE_DIR / path
    return path


def load_actor_samples(task: Task, actor_sample_path: str = 'actor_samples'):
    actor_dir = _resolve_sample_dir(actor_sample_path)
    actor_task_path = actor_dir / f'{task}.json'
    with actor_task_path.open('r') as f:
        return json.load(f)


def load_critic_samples(task: Task, critic_sample_path: str = 'critic_samples'):
    critic_dir = _resolve_sample_dir(critic_sample_path)
    critic_task_path = critic_dir / f'{task}.json'
    with critic_task_path.open('r') as f:
        return json.load(f)


def task_to_command(task: Task, target_vehicle: int | None = None):
    match task:
        case Task.OVERTAKE:
            assert target_vehicle is not None, "Target vehicle must be specified for overtaking task"

            if target_vehicle is None:
                task_language = 'Overtake the car in front of ego vehicle'
            else:
                task_language = f'Overtake car {target_vehicle}'
        
        case Task.GIVE_WAY:
            assert target_vehicle is not None, "Target vehicle must be specified for give way task"

            if target_vehicle is None:
                task_language = 'Give way to the car behind of ego vehicle'
            else:
                task_language = f'Give way to car {target_vehicle}'
        
        case Task.PULL_OVER:
            task_language = 'Safely pull over and stop on the shoulder'
        
        case Task.FOLLOW_LANE:
            task_language = 'Follow current lane'
        
        case Task.CHANGE_LANE_LEFT:
            task_language = 'Change lane to the left'
        
        case Task.CHANGE_LANE_RIGHT:
            task_language = 'Change lane to the right'
        
        case _:
            raise ValueError(f"Unhandled task: {task}")
    
    return task_language


def make_actor_prompt(
    task: Task, command: str,
    actor_sample_path: str = 'actor_samples'
) -> str:
    actor_samples = load_actor_samples(task, actor_sample_path)
    actor_dir = _resolve_sample_dir(actor_sample_path)
    actor_api_path = actor_dir / 'api.txt'
    with actor_api_path.open('r') as f:
        api = f.read()

    actor_prompt_path = actor_dir / 'prompt.txt'
    with actor_prompt_path.open('r') as f:
        actor_prompt = f.read().strip().replace('\n\n', '\n')
    
    actor_prompt = (
        actor_prompt
        .strip()
        .replace('\n\n', '\n')
        .format(
            actor_schema = actor_schema,
            actor_samples = '\n'.join([
                f'{i+1}. {sample}' for i, sample in enumerate(actor_samples)
            ]),
            api = api,
            command = command
        )
    )
    
    return actor_prompt


def make_critic_prompt(
    task: Task, command: str,
    ego_idx: int,
    target_idx: int | None = None,
    critic_sample_path: str = 'critic_samples'
) -> str:
    task_language = task_to_command(task, target_idx)
    critic_samples = load_critic_samples(task, critic_sample_path)
    critic_dir = _resolve_sample_dir(critic_sample_path)

    critic_prompt_path = critic_dir / 'prompt.txt'
    with critic_prompt_path.open('r') as f:
        critic_prompt = f.read().strip().replace('\n\n', '\n')
    
    critic_prompt = (
        critic_prompt
        .strip()
        .replace('\n\n', '\n')
        .format(
            ego_idx = ego_idx,
            target_idx = target_idx,
            task_language = task_language,
            command = command,
            critic_schema = critic_schema,
            critic_samples = '\n'.join([
                f'{i+1}. {sample}' for i, sample in enumerate(critic_samples)
            ])
        )
    )
    
    return critic_prompt


async def query_actor(client: AsyncClient, log: dict,
    limiter: CooldownLimiter,
    actor_sample_path: str = 'actor_samples',
    model_id : str = 'gemini-3.1-flash-lite-preview'
) -> dict[str, str]:
    await limiter.wait()

    task, command = itemgetter(
        'task', 'critic_command'
    )(log)
    actor_prompt = make_actor_prompt(
        task, command, actor_sample_path
    )

    contents = [
        Part(text=actor_prompt)
    ]
    thinking_config = ThinkingConfig(
        include_thoughts=True,
        thinking_budget=-1, # 0 is DISABLED. -1 is AUTOMATIC
    )
    config = GenerateContentConfig(
        response_json_schema = actor_schema,
        response_mime_type = 'application/json',
        thinking_config=thinking_config
    )
    response = await client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config
    )
    response_dict = {}
    for part in response.parts:
        if part.thought:
            response_dict['thinking'] = part.text
        else:
            response_dict['answer'] = part.text

    return response_dict


async def query_critic(client: AsyncClient, log: dict,
    limiter: CooldownLimiter,
    critic_sample_path: str = 'critic_samples',
    model_id : str = 'gemini-3.1-flash-lite-preview'
) -> dict[str, str]:
    await limiter.wait()

    task, command, ego_idx, target_idx, video_path = itemgetter(
        'task', 'command', 'ego_idx', 'target_idx', 'video_path'
    )(log)
    critic_prompt = make_critic_prompt(
        task, command, ego_idx, target_idx, critic_sample_path
    )

    with open(video_path, 'rb') as f:
        video_bytes = open(video_path, 'rb').read()
    
    contents = [
        Part(
            inline_data=Blob(
                data=video_bytes,
                mime_type='video/mp4'),
                video_metadata=VideoMetadata(fps=1)
        ),
        Part(
            text=critic_prompt
        )
    ]
    thinking_config = ThinkingConfig(
        include_thoughts=True,
        thinking_budget=-1, # 0 is DISABLED. -1 is AUTOMATIC
    )
    config = GenerateContentConfig(
        media_resolution = MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        response_json_schema = critic_schema,
        response_mime_type = 'application/json',
        thinking_config=thinking_config
    )
    response = await client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config
    )
    response_dict = {}
    for part in response.parts:
        if part.thought:
            response_dict['thinking'] = part.text
        else:
            response_dict['answer'] = part.text

    return response_dict


async def query_full(
    client: AsyncClient, log: dict,
    limiter: CooldownLimiter,
    critic_sample_path: str = 'critic_samples',
    critic_model_id: str = 'gemini-3.1-flash-lite-preview',
    actor_sample_path: str = 'actor_samples',
    actor_model_id: str = 'gemini-3.1-flash-lite-preview'
) -> dict:
    critic_response = await query_critic(
        client, log, limiter, critic_sample_path, critic_model_id
    )
    log['critic_thinking'] = critic_response['thinking']
    critic_analysis = CriticAnalysis.model_validate_json(
        critic_response['answer']
    )
    log['critic_reason'] = critic_analysis.reason
    log['critic_command'] = critic_analysis.command

    actor_response = await query_actor(
        client, log, limiter, actor_sample_path, actor_model_id
    )
    log['actor_thinking'] = actor_response['thinking']
    actor_code = ActorCode.model_validate_json(
        actor_response['answer']
    )
    log['actor_reason'] = actor_code.reason
    log['actor_code'] = actor_code.code

    return log

async def worker(
    name: str,
    client,
    queue: asyncio.Queue,
    limiter: CooldownLimiter,
    results: list[dict],
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        index, log = item
        try:
            log = await query_full(client, log, limiter)
            results[index] = log
            # print(f"{name}: finished log {index}")
        except Exception as e:
            results[index] = {'Failed': f"{type(e).__name__}: {e}"}
            print(f"{name}: failed log {index}: {e}")
        finally:
            queue.task_done()