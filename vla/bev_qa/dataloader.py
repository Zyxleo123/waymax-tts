from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_QA_DIR = "/zfsauton/scratch/mineuih/waymax_rs/qa_dataset/"
DEFAULT_IMG_ROOT = "/zfsauton/scratch/eshau/imgs_past/"


def _extract_tfrecord_index(path: Path) -> int:
    match = re.search(r"tfrecord-(\d+)-of-01000", path.name)
    if match is None:
        raise ValueError(f"Could not parse tfrecord index from: {path}")
    return int(match.group(1))


def _format_float(value: Any, ndigits: int = 2) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def _answer_to_text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _build_qas_from_answers(answers: dict[str, Any]) -> list[dict[str, str]]:
    """
    Converts one scenario's answer dict into one or more natural-language QA pairs.

    We intentionally skip target_idx because it is an internal object index, not a
    semantic visual answer.
    """
    qas: list[dict[str, str]] = []

    templates = {
        "has_left_lane": "Is there a lane on the left of the ego vehicle? Answer yes or no.",
        "has_right_lane": "Is there a lane on the right of the ego vehicle? Answer yes or no.",
        "num_vehicle_left": "How many vehicles are on the left side of the ego vehicle?",
        "num_vehicle_right": "How many vehicles are on the right side of the ego vehicle?",
        "num_vehicle_front_same_lane": "How many vehicles are in front of the ego vehicle in the same lane?",
        "num_vehicle_behind_same_lane": "How many vehicles are behind the ego vehicle in the same lane?",
        "num_pedestrian_front": "How many pedestrians are in front of the ego vehicle?",
        "traffic_light_state": "What is the traffic light state class? Answer with the integer class.",
        "target_speed": "What is the target object's speed?",
        "target_heading": "What is the target object's heading?",
        "target_type": "What is the target object type?",
    }

    for key, question in templates.items():
        if key in answers:
            qas.append(
                {
                    "key": key,
                    "question": question,
                    "answer": _answer_to_text(answers[key]),
                }
            )

    if "target_x" in answers and "target_y" in answers:
        qas.append(
            {
                "key": "target_position",
                "question": "What is the target object's position relative to the ego vehicle? Answer as x y.",
                "answer": f"{_format_float(answers['target_x'])} {_format_float(answers['target_y'])}",
            }
        )

    return qas


@dataclass(frozen=True)
class BEVQASample:
    image_path: Path
    question: str
    answer: str
    qa_key: str
    tfrecord_index: int
    scenario_id: int
    timestep: int


class BEVQADataset(Dataset):
    def __init__(
        self,
        qa_dir: str = DEFAULT_QA_DIR,
        img_root: str = DEFAULT_IMG_ROOT,
        file_indices: list[int] | None = None,
        timestep: int = 10,
        one_qa_per_scenario: bool = True,
        seed: int = 0,
        max_samples: int | None = None,
    ) -> None:
        self.qa_dir = Path(qa_dir)
        self.img_root = Path(img_root)
        self.timestep = int(timestep)
        self.one_qa_per_scenario = bool(one_qa_per_scenario)
        self.rng = random.Random(seed)

        if file_indices is None:
            qa_files = sorted(self.qa_dir.glob("training_tfexample.tfrecord-*-of-01000.json"))
        else:
            qa_files = [
                self.qa_dir / f"training_tfexample.tfrecord-{idx:05d}-of-01000.json"
                for idx in file_indices
            ]

        samples: list[BEVQASample] = []
        missing_images = 0
        skipped_no_qas = 0

        pbar = tqdm(qa_files, desc="Loading BEV-QA data", unit="file")
        for qa_path in pbar:
            if not qa_path.exists():
                continue

            tf_idx = _extract_tfrecord_index(qa_path)

            with open(qa_path, "r") as f:
                data = json.load(f)

            for scenario_id_str, item in data.items():
                scenario_id = int(scenario_id_str)
                item_timestep = int(item.get("timestep", self.timestep))

                if item_timestep != self.timestep:
                    continue

                answers = item.get("answers", {})
                qas = _build_qas_from_answers(answers)

                if not qas:
                    skipped_no_qas += 1
                    continue

                img_path = (
                    self.img_root
                    / f"tfrecord_{tf_idx}"
                    / f"scenario_{scenario_id}"
                    / f"past_tfrecord_{tf_idx}_scenario_{scenario_id}_start_10_frame_9_hist_0_9.png"
                )

                if not img_path.exists():
                    missing_images += 1
                    continue

                if self.one_qa_per_scenario:
                    qas = [self.rng.choice(qas)]

                for qa in qas:
                    samples.append(
                        BEVQASample(
                            image_path=img_path,
                            question=qa["question"],
                            answer=qa["answer"],
                            qa_key=qa["key"],
                            tfrecord_index=tf_idx,
                            scenario_id=scenario_id,
                            timestep=item_timestep,
                        )
                    )

            pbar.set_postfix(
                samples=len(samples),
                missing=missing_images,
                skipped=skipped_no_qas,
                refresh=False,
            )

        if max_samples is not None:
            samples = samples[: int(max_samples)]

        self.samples = samples
        self.missing_images = missing_images
        self.skipped_no_qas = skipped_no_qas

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No BEV-QA samples found. qa_dir={self.qa_dir}, img_root={self.img_root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")

        return {
            "image": image,
            "question": sample.question,
            "answer": sample.answer,
            "qa_key": sample.qa_key,
            "image_path": str(sample.image_path),
            "tfrecord_index": sample.tfrecord_index,
            "scenario_id": sample.scenario_id,
            "timestep": sample.timestep,
        }


def bev_qa_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "qa_keys": [item["qa_key"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "tfrecord_indices": [item["tfrecord_index"] for item in batch],
        "scenario_ids": [item["scenario_id"] for item in batch],
        "timesteps": [item["timestep"] for item in batch],
    }


def build_bev_qa_dataloader(
    qa_dir: str = DEFAULT_QA_DIR,
    img_root: str = DEFAULT_IMG_ROOT,
    file_indices: list[int] | None = None,
    batch_size: int = 2,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_samples: int | None = None,
    one_qa_per_scenario: bool = True,
) -> DataLoader:
    dataset = BEVQADataset(
        qa_dir=qa_dir,
        img_root=img_root,
        file_indices=file_indices,
        one_qa_per_scenario=one_qa_per_scenario,
        seed=seed,
        max_samples=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=bev_qa_collate_fn,
    )


# if __name__ == "__main__":
#     dataset = BEVQADataset()
#     print("num samples:", len(dataset))
#     print("missing images:", dataset.missing_images)
#     print("skipped no qas:", dataset.skipped_no_qas)

#     item = dataset[0]
#     print("\nfirst item:")
#     for key, value in item.items():
#         if key == "image":
#             print(key, value.size)
#         else:
#             print(key, value)
