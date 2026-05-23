#!/usr/bin/env python3
"""
Build byte-offset indices from gemma4 JSONL files, generate instructions
from trajectory summaries using a causal LM, and save per-tfrecord
JSONL files plus index files for fast random access.

Usage (example):
  python language/generate_instructions.py \
      --input-glob '/zfsauton/scratch/eshau/gemma4_31b_output/gemma4_waymax_structured_full_imgs_s*.jsonl' \
      --out-dir /zfsauton/scratch/mineuih/waymax_rs/instructions/training \
      --timesteps 10,20,30,40 --dry-run
"""
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import sys
from typing import List, Dict, Tuple
os.environ["LD_LIBRARY_PATH"] = "/usr/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LIBRARY_PATH"] = "/usr/lib64:" + os.environ.get("LIBRARY_PATH", "")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="/zfsauton/scratch/eshau/gemma4_31b_output", help="Directory for input gemma4 jsonl files")
    # p.add_argument("--out_dir", default="/zfsauton/scratch/mineuih/waymax_rs/instructions/training", help="Directory to write output instruction jsonl files")
    p.add_argument("--out_dir", default="/zfsauton/scratch/mineuih/waymax_rs/instructions/training_qwen", help="Directory to write output instruction jsonl files")
    p.add_argument("--timesteps", default="10,20,30,40", help="Comma-separated timesteps to extract")
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default='cuda')
    p.add_argument("--dry-run", action="store_true", help="Do not call model.generate, store summaries only")
    p.add_argument("--limit-files", type=int, default=0, help="Limit number of input files (for testing)")
    p.add_argument("--skip-index-build", action="store_true", help="Assume indices already built (not used currently)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    return p.parse_args()


def make_prompt(behavior_summary: str) -> str:
    prompt = f"""Convert the following ego vehicle behavior summary into a short driving instruction for a Vision Language Agent (VLA).
The instruction should be concise and direct, starting with an action verb.
Output only the instruction in one sentence, without any additional explanation or context.
example: "Ego vehicle 0 continues straight forward along its visible future trajectory." -> "Go straight forward."
If the summary is ambiguous or does not provide enough information to generate a clear instruction, respond with "Unclear".

Note: The provided `behavior_summary` may have been generated from an image that included a dashed-line overlay indicating the ego vehicle's future trajectory.
When writing the instruction, assume the Vision Language Agent cannot see any dashed-line overlays or trajectory guides.
Do not reference dashed lines, overlays, or explicit trajectory graphics in the instruction—describe only observable actions, maneuvers, or goals.
example: "Ego vehicle 0 follows a dashed blue path that curves slightly to the right." -> "Go slightly to the right while moving forward."
Behavior summary: {behavior_summary}
Driving instruction:"""

    return prompt


def build_file_list(input_dir: str, limit_files: int = 0) -> List[str]:
    from glob import glob
    files = sorted(glob(os.path.join(input_dir, "*.jsonl")))
    if limit_files and limit_files > 0:
        files = files[:limit_files]
    return files


def build_indices(file_paths: List[str], timesteps: List[int]):
    """Scan input files in binary, record byte offsets for matching timesteps.

    Returns: indices: dict[tfrecord_i][scenario_i][timestep] -> list of (file_idx, byte_offset)
    """
    indices: Dict[int, Dict[int, Dict[int, List[Tuple[int, int]]]]] = {}
    for i, file_path in enumerate(file_paths):
        p = Path(file_path)
        if not p.exists():
            print(f"Warning: {file_path} not found, skipping", file=sys.stderr)
            continue
        with p.open("rb") as f:
            offset = 0
            for raw in tqdm(f, desc=f"Scanning {p.name}"):
                line = raw.decode("utf-8")
                try:
                    data = json.loads(line)
                except Exception:
                    offset += len(raw)
                    continue
                try:
                    png_name = os.path.basename(data.get("image_path", ""))
                    parts = png_name.split("_")
                    # expected format somewhere like: prefix_{tfrecord}_x_{scenario}_y_{start}.png
                    tfrecord_i = int(parts[1])
                    scenario_i = int(parts[3])
                    start_i = int(parts[5].split('.')[0])
                except Exception:
                    offset += len(raw)
                    continue

                if start_i in timesteps:
                    indices.setdefault(tfrecord_i, {}).setdefault(scenario_i, {}).setdefault(start_i, []).append((i, offset))

                offset += len(raw)
    return indices


def init_llm(model_name: str):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
        repetition_penalty=1.2,
    )
    return llm, sampling_params


def process_single_tfrecord(args_tuple):
    """Process a single tfrecord."""
    tfrecord_i, indices, file_paths, out_dir, model_name, dry_run, llm, sampling_params, batch_size = args_tuple
    
    if not dry_run and llm is None:
        llm, sampling_params = init_llm(model_name)

    out_file = Path(out_dir) / f"tfrecord_{tfrecord_i}.jsonl"
    out_index = {}
    
    # Collect all records for this tfrecord
    all_records = []
    all_prompts = []
    
    for scenario_i in sorted(indices[tfrecord_i].keys()):
        for timestep in sorted(indices[tfrecord_i][scenario_i].keys()):
            hits = indices[tfrecord_i][scenario_i][timestep]
            for file_idx, byte_offset in hits:
                in_path = Path(file_paths[file_idx])
                with in_path.open("rb") as inf:
                    inf.seek(byte_offset)
                    raw = inf.readline()
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except Exception as e:
                        continue
                    parsed = obj.get("parsed", {})
                    trajectory_summary = parsed.get("trajectory_summary")
                    risks = parsed.get("risks")

                    if trajectory_summary is None:
                        continue

                    record_base = {
                        "scenario_index": scenario_i,
                        "timestep": timestep,
                        "summary": trajectory_summary,
                        "risks": risks,
                    }
                    
                    if dry_run:
                        record_base["instruction"] = ""
                        all_records.append(record_base)
                    else:
                        prompt = make_prompt(trajectory_summary)
                        all_records.append(record_base)
                        all_prompts.append(prompt)
    
    # Generate instructions in batches
    if not dry_run and all_prompts:
        instructions = []
        for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"tfrecord {tfrecord_i} inference"):
            batch_prompts = all_prompts[i:i + batch_size]
            ptr = i + batch_size
            try:
                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                for output in outputs:
                    instruction = output.outputs[0].text.strip()
                    instructions.append(instruction.split("\n")[0].split(".")[0].strip())  # take only the first sentence if multiple
            except Exception as e:
                print(f"Error during inference: {e}", file=sys.stderr)
                instructions.extend([""] * len(batch_prompts))
        batch_prompts = all_prompts[ptr:]
        if batch_prompts:
            try:
                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                for output in outputs:
                    instruction = output.outputs[0].text.strip()
                    instructions.append(instruction.split("\n")[0].split(".")[0].strip())  # take only the first sentence if multiple
            except Exception as e:
                print(f"Error during inference: {e}", file=sys.stderr)
                instructions.extend([""] * len(batch_prompts))
        
        # Assign instructions back to records
        for idx, record in enumerate(all_records):
            if idx < len(instructions):
                record["instruction"] = instructions[idx]
            else:
                record["instruction"] = ""
    
    # Write to file
    with out_file.open("wb") as out_f:
        for record in all_records:
            rec_bytes = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            cur_offset = out_f.tell()
            out_f.write(rec_bytes)
            key = f"{record['scenario_index']}:{record['timestep']}"
            out_index.setdefault(key, []).append(cur_offset)

    # write index file
    idx_path = Path(out_dir) / f"tfrecord_{tfrecord_i}_index.json"
    with idx_path.open("w", encoding="utf-8") as idxf:
        json.dump(out_index, idxf)
    
    return f"Wrote tfrecord_{tfrecord_i}"


def write_tfrecord_outputs(indices, file_paths, out_dir: str, model_name: str, device: str = None, dry_run: bool = False, batch_size: int = 16):
    """Write tfrecord outputs using a single process/GPU."""
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    llm = None
    sampling_params = None
    if not dry_run:
        llm, sampling_params = init_llm(model_name)
    for tfrecord_i in sorted(indices.keys()):
        task_args = (tfrecord_i, indices, file_paths, out_dir, model_name, dry_run, llm, sampling_params, batch_size)
        tasks.append(task_args)

    print(f"Processing {len(tasks)} tfrecords on a single GPU")

    for task_args in tasks:
        tfrecord_i = task_args[0]
        out_file = Path(out_dir) / f"tfrecord_{tfrecord_i}.jsonl"
        if out_file.exists():
            continue
        else:
            # save dummy file to indicate in-progress work
            out_file.touch()
        result = process_single_tfrecord(task_args)
        print(result)


def main():
    args = parse_args()
    timesteps = [int(x) for x in args.timesteps.split(",") if x.strip()]
    file_paths = build_file_list(args.input_dir, args.limit_files)
    if not file_paths:
        print("No input files found for the provided glob", file=sys.stderr)
        return

    print(f"Found {len(file_paths)} input files; timesteps={timesteps}")
    print(f"Using backend: vllm, batch_size={args.batch_size}")
    indices = build_indices(file_paths, timesteps)

    if not indices:
        print("No matching records found for requested timesteps.")
        return

    write_tfrecord_outputs(
        indices, 
        file_paths, 
        args.out_dir, 
        args.model_name, 
        device=args.device, 
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
