import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 설정
model_name = "google/gemma-4-E2B-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")

# 모델 로드 (최적화됨)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# BOS 토큰 설정
tokenizer.add_bos_token = True

print("Model loaded successfully!")
print(f"BOS token: {tokenizer.bos_token_id}, EOS token: {tokenizer.eos_token_id}")

def generate(prompt: str, max_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
    full_prompt = f"<bos>{prompt}" 
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # 반복 제약
            no_repeat_ngram_size=2,  # 2-gram 반복 금지
            length_penalty=1.0
        )

    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text.strip()

def make_prompt(behavior_summary: str) -> str:
    prompt = f"""Convert the following ego vehicle behavior summary into a short driving instruction for a Vision Language Agent (VLA).
The instruction should be concise and direct, starting with an action verb.
Output only the instruction without any additional explanation or context.
example: "Ego vehicle 0 continues straight forward along its visible future trajectory." -> "Go straight forward."
If the summary is ambiguous or does not provide enough information to generate a clear instruction, respond with "Unclear".

Note: The provided `behavior_summary` may have been generated from an image that included a dashed-line overlay indicating the ego vehicle's future trajectory (for example, phrases like "ego vehicle follows dashed line"). 
When writing the instruction, assume the Vision Language Agent cannot see any dashed-line overlays or trajectory guides. 
Do not reference dashed lines, overlays, or explicit trajectory graphics in the instruction—describe only observable actions, maneuvers, or goals.
example: "Ego vehicle 0 follows a dashed blue path that curves slightly to the right." -> "Go slightly to the right while moving forward."
Behavior summary: {behavior_summary}
Driving instruction:"""
    return prompt

if __name__ == "__main__":
    import jsonlines
    import json
    from collections import OrderedDict
    from tqdm import tqdm
    with open("/zfsauton/scratch/eshau/query_waymax_outputs/gemma4_waymax_structured_best_50k_merged.jsonl") as f:
        data = list(jsonlines.Reader(f))
    file_path = "/zfsauton/scratch/mineuih/waymax_rs/gemma4_50k_merged_instructions.jsonl"
    
    for i, item in enumerate(tqdm(data)):
        image_path = item['image_path'].split("/")[-1]
        tfrecord = int(image_path.split("_")[1])
        scenario = int(image_path.split("_")[3])
        start = int(image_path.split("_")[5])
        traj_summary = item['parsed']['trajectory_summary']
        prompt = make_prompt(traj_summary)
        generated_instruction = generate(prompt)
        result = OrderedDict({
            "tfrecord": tfrecord,
            "scenario": scenario,
            "start": start,
            "summary": traj_summary,
            "instruction": generated_instruction
        })
        if i == 0:
            with open(file_path, "w") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
        else:
            with open(file_path, "a") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")