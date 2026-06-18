import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import io
import json
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, "../waymo-open-dataset/src")
from waymo_open_dataset.protos import scenario_pb2
print("import success:", scenario_pb2.__file__)

@dataclass
class LaneNode:
    lane_feature_id: int
    point_index: int
    xyz: np.ndarray  # shape (3,)

@dataclass
class LaneGraph:
    scenario_id: str
    nodes_xyz: np.ndarray                  # [N, 3]
    node_lane_ids: np.ndarray             # [N]
    node_point_indices: np.ndarray        # [N]
    polyline_edges: np.ndarray            # [E1, 2]  (within-lane consecutive points)
    successor_edges: np.ndarray           # [E2, 2]  (last node of lane A -> first node of lane B)
    predecessor_edges: np.ndarray         # [E3, 2]
    left_neighbor_edges: np.ndarray       # [E4, 2]  (optional, lane-level approximated as end->start)
    right_neighbor_edges: np.ndarray      # [E5, 2]
    lane_id_to_node_range: Dict[int, Tuple[int, int]]  # lane_id -> [start_node_idx, end_node_idx_exclusive]

def _bytes_feature_to_str(feature: tf.train.Feature) -> str:
    if feature.bytes_list.value:
        return feature.bytes_list.value[0].decode("utf-8")
    raise KeyError("Feature is not a bytes feature or is empty.")


def _int64_feature_to_int(feature: tf.train.Feature) -> int:
    if feature.int64_list.value:
        return int(feature.int64_list.value[0])
    raise KeyError("Feature is not an int64 feature or is empty.")

def get_feature_array(feature):
    """tf.train.Feature -> numpy array"""
    if len(feature.float_list.value) > 0:
        return np.array(feature.float_list.value, dtype=np.float32)
    if len(feature.int64_list.value) > 0:
        return np.array(feature.int64_list.value, dtype=np.int64)
    if len(feature.bytes_list.value) > 0:
        return np.array(feature.bytes_list.value)
    return None


def extract_tfexample_roadgraph_points(ex):
    feats = ex.features.feature


    # 1) xyz 한 번에 저장된 경우
    key = "roadgraph_samples/xyz"
    if key in feats:
        arr = get_feature_array(feats[key])
        arr = np.asarray(arr, dtype=np.float32)
        valid = get_feature_array(feats.get("roadgraph_samples/valid"))

        if arr.size % 3 != 0:
            raise ValueError(f"{key} size={arr.size} is not divisible by 3")

        return arr.reshape(-1, 3)[valid.astype(bool)]
    available = sorted(feats.keys())
    raise KeyError(
        "Could not find tfexample roadgraph point keys. "
        f"Available keys include: {available[:100]}"
    )

def get_scenario_id_from_tfexample_bytes(example_bytes: bytes) -> str:
    """
    Try several likely feature names for scenario id.
    Different preprocessing pipelines may use different names.
    """
    ex = tf.train.Example()
    ex.ParseFromString(example_bytes)
    feats = ex.features.feature

    candidate_keys = [
        "scenario/id",
        "scenario_id",
        "state/scenario_id",
        "id",
    ]

    for key in candidate_keys:
        if key in feats:
            f = feats[key]
            # Usually bytes, sometimes int64/string-like
            if f.bytes_list.value:
                return _bytes_feature_to_str(f)
            if f.int64_list.value:
                return str(_int64_feature_to_int(f))

    available = list(feats.keys())[:50]
    raise KeyError(
        "Could not find scenario id in tf.Example. "
        f"Tried keys={candidate_keys}. "
        f"Available example keys (first 50)={available}"
    )


# -----------------------------
# Helpers: inspect map feature type
# -----------------------------
def get_map_feature_type_name(map_feature: Any) -> Optional[str]:
    """
    Returns the active oneof field name for Scenario.map_features item.
    Common values include: lane, road_line, road_edge, crosswalk, speed_bump, stop_sign, driveway
    depending on WOMD version.
    """
    # Newer protobuf API supports WhichOneof
    try:
        return map_feature.WhichOneof("feature_data")
    except Exception:
        pass

    # Fallback: heuristic
    candidate_fields = [
        "lane",
        "road_line",
        "road_edge",
        "crosswalk",
        "speed_bump",
        "stop_sign",
        "driveway",
    ]
    for name in candidate_fields:
        try:
            field = getattr(map_feature, name)
            # Heuristic: serialized size > 0 means present
            if field is not None and field.ByteSize() > 0:
                return name
        except Exception:
            continue
    return None


def polyline_to_xyz(polyline) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in polyline], dtype=np.float32)
    return pts


# -----------------------------
# Scenario -> lane graph
# -----------------------------
def build_lane_graph_from_scenario(
    scenario: scenario_pb2.Scenario,
    include_lane_relations: bool = True,
) -> LaneGraph:
    """
    Builds a point-level lane graph from Scenario.map_features.

    Node:
      each point in each lane center polyline

    Edges:
      - polyline_edges: consecutive points within a lane
      - successor_edges / predecessor_edges:
          if lane relation fields exist, connect lane end -> successor start
      - left/right_neighbor_edges:
          approximated lane-level relation edge using lane start nodes
          (because neighbor relation is lane-level, not point-level)
    """
    scenario_id = scenario.scenario_id

    nodes_xyz_list: List[np.ndarray] = []
    node_lane_ids_list: List[np.ndarray] = []
    node_point_indices_list: List[np.ndarray] = []
    polyline_edges: List[Tuple[int, int]] = []

    lane_id_to_node_range: Dict[int, Tuple[int, int]] = {}
    lane_feature_map: Dict[int, Any] = {}

    # 1) collect lane polylines
    global_node_offset = 0
    for mf in scenario.map_features:
        feature_type = get_map_feature_type_name(mf)
        if feature_type != "lane":
            continue

        lane = mf.lane
        lane_id = int(mf.id)
        lane_feature_map[lane_id] = lane

        pts = polyline_to_xyz(lane.polyline)
        if len(pts) == 0:
            continue

        n = len(pts)
        start = global_node_offset
        end = global_node_offset + n

        lane_id_to_node_range[lane_id] = (start, end)
        nodes_xyz_list.append(pts)
        node_lane_ids_list.append(np.full((n,), lane_id, dtype=np.int64))
        node_point_indices_list.append(np.arange(n, dtype=np.int64))

        for i in range(start, end - 1):
            polyline_edges.append((i, i + 1))

        global_node_offset = end

    if not nodes_xyz_list:
        return LaneGraph(
            scenario_id=scenario_id,
            nodes_xyz=np.zeros((0, 3), dtype=np.float32),
            node_lane_ids=np.zeros((0,), dtype=np.int64),
            node_point_indices=np.zeros((0,), dtype=np.int64),
            polyline_edges=np.zeros((0, 2), dtype=np.int64),
            successor_edges=np.zeros((0, 2), dtype=np.int64),
            predecessor_edges=np.zeros((0, 2), dtype=np.int64),
            left_neighbor_edges=np.zeros((0, 2), dtype=np.int64),
            right_neighbor_edges=np.zeros((0, 2), dtype=np.int64),
            lane_id_to_node_range={},
        )

    nodes_xyz = np.concatenate(nodes_xyz_list, axis=0)
    node_lane_ids = np.concatenate(node_lane_ids_list, axis=0)
    node_point_indices = np.concatenate(node_point_indices_list, axis=0)

    successor_edges: List[Tuple[int, int]] = []
    predecessor_edges: List[Tuple[int, int]] = []
    left_neighbor_edges: List[Tuple[int, int]] = []
    right_neighbor_edges: List[Tuple[int, int]] = []

    # 2) lane-to-lane relations if available in the proto
    if include_lane_relations:
        for lane_id, lane in lane_feature_map.items():
            src_range = lane_id_to_node_range.get(lane_id)
            if src_range is None:
                continue

            src_start, src_end = src_range
            src_first = src_start
            src_last = src_end - 1

            # successor lanes
            if hasattr(lane, "exit_lanes"):
                for succ_lane_id in lane.exit_lanes:
                    succ_lane_id = int(succ_lane_id)
                    dst_range = lane_id_to_node_range.get(succ_lane_id)
                    if dst_range is not None:
                        dst_first = dst_range[0]
                        successor_edges.append((src_last, dst_first))

            # predecessor lanes
            if hasattr(lane, "entry_lanes"):
                for pred_lane_id in lane.entry_lanes:
                    pred_lane_id = int(pred_lane_id)
                    pred_range = lane_id_to_node_range.get(pred_lane_id)
                    if pred_range is not None:
                        pred_last = pred_range[1] - 1
                        predecessor_edges.append((pred_last, src_first))

            # left neighbors
            # Depending on WOMD version, neighbor fields can differ.
            # We try a few common patterns.
            left_neighbor_lane_ids = []

            if hasattr(lane, "left_neighbors"):
                for nb in lane.left_neighbors:
                    if hasattr(nb, "feature_id"):
                        left_neighbor_lane_ids.append(int(nb.feature_id))
                    elif hasattr(nb, "lane_id"):
                        left_neighbor_lane_ids.append(int(nb.lane_id))

            for nb_lane_id in left_neighbor_lane_ids:
                dst_range = lane_id_to_node_range.get(nb_lane_id)
                if dst_range is not None:
                    # lane-level approximate relation edge
                    left_neighbor_edges.append((src_first, dst_range[0]))

            # right neighbors
            right_neighbor_lane_ids = []

            if hasattr(lane, "right_neighbors"):
                for nb in lane.right_neighbors:
                    if hasattr(nb, "feature_id"):
                        right_neighbor_lane_ids.append(int(nb.feature_id))
                    elif hasattr(nb, "lane_id"):
                        right_neighbor_lane_ids.append(int(nb.lane_id))

            for nb_lane_id in right_neighbor_lane_ids:
                dst_range = lane_id_to_node_range.get(nb_lane_id)
                if dst_range is not None:
                    right_neighbor_edges.append((src_first, dst_range[0]))

    return LaneGraph(
        scenario_id=scenario_id,
        nodes_xyz=nodes_xyz,
        node_lane_ids=node_lane_ids,
        node_point_indices=node_point_indices,
        polyline_edges=np.asarray(polyline_edges, dtype=np.int64).reshape(-1, 2),
        successor_edges=np.asarray(successor_edges, dtype=np.int64).reshape(-1, 2),
        predecessor_edges=np.asarray(predecessor_edges, dtype=np.int64).reshape(-1, 2),
        left_neighbor_edges=np.asarray(left_neighbor_edges, dtype=np.int64).reshape(-1, 2),
        right_neighbor_edges=np.asarray(right_neighbor_edges, dtype=np.int64).reshape(-1, 2),
        lane_id_to_node_range=lane_id_to_node_range,
    )


# -----------------------------
# Build index from original Scenario TFRecord
# -----------------------------
def build_scenario_lane_graph_index(
    scenario_tfrecord_paths: List[str],
    max_scenarios: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, LaneGraph]:
    """
    Reads original Scenario TFRecords and builds:
      scenario_id -> LaneGraph
    """
    graph_index: Dict[str, LaneGraph] = {}
    num = 0

    for path in scenario_tfrecord_paths:
        ds = tf.data.TFRecordDataset(path)
        for raw in ds:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw.numpy())

            graph = build_lane_graph_from_scenario(scenario)
            graph_index[graph.scenario_id] = graph

            num += 1
            if verbose and num % 100 == 0:
                print(f"[build_index] processed {num} scenarios")

            if max_scenarios is not None and num >= max_scenarios:
                return graph_index

    return graph_index


def iterate_tfexamples_with_lane_graph(
    tfexample_paths,
    lane_graph_index,
    max_examples=None,
):
    num_matched = 0
    num_total = 0
    num_unmatched = 0

    for path in tfexample_paths:
        ds = tf.data.TFRecordDataset(path)
        for raw in ds:
            num_total += 1
            ex_bytes = raw.numpy()
            scenario_id = get_scenario_id_from_tfexample_bytes(ex_bytes)

            if scenario_id not in lane_graph_index:
                num_unmatched += 1
                if num_unmatched <= 10:
                    print(f"[unmatched] scenario_id={repr(scenario_id)}")
                continue

            ex = tf.train.Example()
            ex.ParseFromString(ex_bytes)
            print(f"[matched] scenario_id={repr(scenario_id)}")
            yield scenario_id, ex, lane_graph_index[scenario_id]

            num_matched += 1
            if max_examples is not None and num_matched >= max_examples:
                print(f"total={num_total}, matched={num_matched}, unmatched={num_unmatched}")
                return

    print(f"total={num_total}, matched={num_matched}, unmatched={num_unmatched}")


def lane_graph_to_npz_bytes(graph: LaneGraph) -> bytes:
    """Serialize one LaneGraph into NPZ bytes (pickle-free)."""
    lane_items = sorted(graph.lane_id_to_node_range.items(), key=lambda x: x[0])
    lane_ids = np.array([k for k, _ in lane_items], dtype=np.int64)
    lane_starts = np.array([v[0] for _, v in lane_items], dtype=np.int64)
    lane_ends = np.array([v[1] for _, v in lane_items], dtype=np.int64)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        schema_version=np.array(["v1"]),
        scenario_id=np.array([graph.scenario_id]),
        nodes_xyz=graph.nodes_xyz,
        node_lane_ids=graph.node_lane_ids,
        node_point_indices=graph.node_point_indices,
        polyline_edges=graph.polyline_edges,
        successor_edges=graph.successor_edges,
        predecessor_edges=graph.predecessor_edges,
        left_neighbor_edges=graph.left_neighbor_edges,
        right_neighbor_edges=graph.right_neighbor_edges,
        lane_ids=lane_ids,
        lane_starts=lane_starts,
        lane_ends=lane_ends,
    )
    return buf.getvalue()


def collect_tfexample_scenario_ids(tfexample_path: str):
    """Read one tfexample shard and return ordered scenario_ids."""
    scenario_ids = []
    ds = tf.data.TFRecordDataset(tfexample_path)
    for raw in ds:
        ex_bytes = raw.numpy()
        sid = get_scenario_id_from_tfexample_bytes(ex_bytes)
        scenario_ids.append(sid)
    return scenario_ids


def save_lanegraph_shard_zip(
    output_zip_path: str,
    tfexample_path: str,
    scenario_ids,
    lanegraphs,
    verbose: bool = True,
):
    """
    Save one file per tfexample shard.
    Each entry inside the zip is one scenario LaneGraph NPZ and its index preserves tfexample order.
    """
    if len(scenario_ids) != len(lanegraphs):
        raise ValueError("scenario_ids and lanegraphs length mismatch")

    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

    manifest = {
        "schema_version": "lanegraph_shard_v1",
        "tfexample_file": os.path.basename(tfexample_path),
        "num_records": len(scenario_ids),
        "scenario_ids": scenario_ids,
        "missing_indices": [i for i, g in enumerate(lanegraphs) if g is None],
    }

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        for i, (sid, graph) in enumerate(zip(scenario_ids, lanegraphs)):
            if graph is None:
                continue
            npz_bytes = lane_graph_to_npz_bytes(graph)
            zf.writestr(f"{i:06d}_{sid}.npz", npz_bytes)

    if verbose:
        num_saved = sum(g is not None for g in lanegraphs)
        print(
            f"[save] {output_zip_path} | total={len(scenario_ids)} saved={num_saved} missing={len(manifest['missing_indices'])}"
        )


def build_and_save_lanegraph_shards_by_tfexample(
    tfexample_files,
    scenario_files,
    output_dir: str,
    verbose: bool = True,
):
    """
    Pipeline:
      1) For each tfexample shard, collect ordered scenario ids
      2) Iterate scenario shards and build LaneGraph only for matched scenario ids
      3) Save one lanegraph zip per tfexample shard
         - order matches tfexample record order
    """
    # Step 1: Build targets from tfexample shards
    shard_scenario_ids = []
    scenario_to_targets = {}  # scenario_id -> list[(shard_idx, record_idx)]

    for shard_idx, tf_path in enumerate(tqdm(tfexample_files, desc="Collecting scenario ids from tfexample shards")):
        ids = collect_tfexample_scenario_ids(tf_path)
        shard_scenario_ids.append(ids)
        for rec_idx, sid in enumerate(ids):
            scenario_to_targets.setdefault(sid, []).append((shard_idx, rec_idx))

        if verbose:
            print(f"[tfexample] shard={shard_idx} records={len(ids)} file={os.path.basename(tf_path)}")

    lanegraph_slots = [[None for _ in ids] for ids in shard_scenario_ids]
    remaining = set(scenario_to_targets.keys())

    # Step 2: Scan scenario shards and fill matched slots
    num_matched = 0
    for sc_path in tqdm(scenario_files, desc="Scanning scenario shards"):
        ds = tf.data.TFRecordDataset(sc_path)
        for raw in ds:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw.numpy())
            sid = scenario.scenario_id

            if sid not in remaining:
                continue

            graph = build_lane_graph_from_scenario(scenario)
            for shard_idx, rec_idx in scenario_to_targets[sid]:
                lanegraph_slots[shard_idx][rec_idx] = graph

            num_matched += 1
            remaining.remove(sid)

        if verbose:
            print(
                f"[scenario_scan] file={os.path.basename(sc_path)} matched_so_far={num_matched} remaining={len(remaining)}"
            )

        if not remaining:
            break

    # Step 3: Save per tfexample shard
    os.makedirs(output_dir, exist_ok=True)
    for shard_idx, tf_path in enumerate(tfexample_files):
        tf_name = os.path.basename(tf_path)
        out_name = f"{tf_name}.lanegraph.zip"
        out_path = os.path.join(output_dir, out_name)
        save_lanegraph_shard_zip(
            output_zip_path=out_path,
            tfexample_path=tf_path,
            scenario_ids=shard_scenario_ids[shard_idx],
            lanegraphs=lanegraph_slots[shard_idx],
            verbose=verbose,
        )

    if verbose:
        print(
            f"[done] shards={len(tfexample_files)} total_unique_targets={len(scenario_to_targets)} unresolved={len(remaining)}"
        )

    return {
        "num_tfexample_shards": len(tfexample_files),
        "num_unique_scenario_ids": len(scenario_to_targets),
        "num_unresolved_scenario_ids": len(remaining),
    }

def main():
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/zfsauton/datasets/womd")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--type", type=str, default="training", choices=["training", "validation", "testing"])
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    num_shards = len(glob(f"{dataset_dir}/scenario/{args.type}/{args.type}.tfrecord-*"))
    partition_size = 50

    scenario_files = [
        f"{dataset_dir}/scenario/{args.type}/{args.type}.tfrecord-{i:05d}-of-{num_shards:05d}"
        for i in range(num_shards)
    ]
    tfexample_files = [
        f"{dataset_dir}/tf_example/{args.type}/{args.type}_tfexample.tfrecord-{i:05d}-of-{num_shards:05d}"
        for i in range(num_shards)
    ]

    for i in range(num_shards // partition_size):
        print(f"making lane graph shards for {i * partition_size} ~ {(i + 1) * partition_size} scenario shards")
        summary = build_and_save_lanegraph_shards_by_tfexample(
            tfexample_files=tfexample_files[i * partition_size : (i + 1) * partition_size],
            scenario_files=scenario_files,
            output_dir=output_dir,
            verbose=False,
        )
        print(summary)

if __name__ == "__main__":
    main()