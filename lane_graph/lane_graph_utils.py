from __future__ import annotations

import io
import json
import math
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from pathlib import Path
import numpy as np


@dataclass
class LaneGraphData:
    scenario_id: str
    nodes_xyz: np.ndarray
    node_lane_ids: np.ndarray
    node_point_indices: np.ndarray
    polyline_edges: np.ndarray
    successor_edges: np.ndarray
    predecessor_edges: np.ndarray
    left_neighbor_edges: np.ndarray
    right_neighbor_edges: np.ndarray
    lane_id_to_node_range: Dict[int, Tuple[int, int]]


def _lane_graph_zip_path_for_tfrecord(tfrecord_path: str, lane_graph_dir: str) -> Path:
    """Maps scenario tfrecord filename to lanegraph shard zip filename."""
    base = os.path.basename(tfrecord_path)
    return Path(lane_graph_dir) / f"{base}.lanegraph.zip"


class LaneGraphZipStore:
    """Small helper that loads lane graph NPZ entries from shard zip files.

    Each shard zip is expected to follow the format created by
    scripts/extract_roadgraph.py (manifest.json + one npz per scenario record).
    """

    def __init__(self, lane_graph_dir: str):
        self.lane_graph_dir = lane_graph_dir
        self._index_built = False
        self._scenario_to_location: Dict[str, Tuple[str, int]] = {}
        self._cache: Dict[str, LaneGraphData] = {}

    def _build_index(self) -> None:
        if self._index_built:
            return

        if not os.path.isdir(self.lane_graph_dir):
            raise FileNotFoundError(f"lane_graph_dir does not exist: {self.lane_graph_dir}")

        for name in sorted(os.listdir(self.lane_graph_dir)):
            if not name.endswith(".lanegraph.zip"):
                continue
            zip_path = os.path.join(self.lane_graph_dir, name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                if "manifest.json" not in zf.namelist():
                    continue
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                scenario_ids = manifest.get("scenario_ids", [])
                for rec_idx, scenario_id in enumerate(scenario_ids):
                    self._scenario_to_location[str(scenario_id)] = (zip_path, int(rec_idx))

        self._index_built = True

    @staticmethod
    def _load_lane_graph_from_npz_bytes(npz_bytes: bytes) -> LaneGraphData:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
            scenario_id = str(data["scenario_id"][0])
            lane_ids = data["lane_ids"].astype(np.int64)
            lane_starts = data["lane_starts"].astype(np.int64)
            lane_ends = data["lane_ends"].astype(np.int64)
            lane_id_to_node_range = {
                int(lid): (int(start), int(end))
                for lid, start, end in zip(lane_ids, lane_starts, lane_ends)
            }
            return LaneGraphData(
                scenario_id=scenario_id,
                nodes_xyz=data["nodes_xyz"].astype(np.float32),
                node_lane_ids=data["node_lane_ids"].astype(np.int64),
                node_point_indices=data["node_point_indices"].astype(np.int64),
                polyline_edges=data["polyline_edges"].astype(np.int64),
                successor_edges=data["successor_edges"].astype(np.int64),
                predecessor_edges=data["predecessor_edges"].astype(np.int64),
                left_neighbor_edges=data["left_neighbor_edges"].astype(np.int64),
                right_neighbor_edges=data["right_neighbor_edges"].astype(np.int64),
                lane_id_to_node_range=lane_id_to_node_range,
            )

    def get(self, scenario_id: str) -> Optional[LaneGraphData]:
        scenario_id = str(scenario_id)
        if scenario_id in self._cache:
            return self._cache[scenario_id]

        self._build_index()
        location = self._scenario_to_location.get(scenario_id)
        if location is None:
            return None

        zip_path, rec_idx = location
        with zipfile.ZipFile(zip_path, "r") as zf:
            prefix = f"{int(rec_idx):06d}_"
            matches = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".npz")]
            if not matches:
                return None
            lane_graph = self._load_lane_graph_from_npz_bytes(zf.read(matches[0]))
            self._cache[scenario_id] = lane_graph
            return lane_graph


class LaneGraphShardStore:
    """Loads lane graphs from a single shard zip using scenario_index.

    The zip is opened once and then reused for repeated lookups.
    """

    def __init__(self, lane_graph_zip_path: str):
        self.lane_graph_zip_path = lane_graph_zip_path
        self._zf: Optional[zipfile.ZipFile] = None
        self._index_built = False
        self._entry_by_record_index: Dict[int, str] = {}
        self._cache_by_record_index: Dict[int, LaneGraphData] = {}

    def _ensure_open(self) -> None:
        if self._zf is None:
            if not os.path.isfile(self.lane_graph_zip_path):
                raise FileNotFoundError(
                    f"lane graph shard zip does not exist: {self.lane_graph_zip_path}"
                )
            self._zf = zipfile.ZipFile(self.lane_graph_zip_path, "r")

    def _build_index(self) -> None:
        if self._index_built:
            return
        self._ensure_open()
        assert self._zf is not None

        for name in self._zf.namelist():
            if not name.endswith(".npz"):
                continue
            prefix = name.split("_", 1)[0]
            try:
                rec_idx = int(prefix)
            except ValueError:
                continue
            self._entry_by_record_index[rec_idx] = name

        self._index_built = True

    def get_by_record_index(self, scenario_index: int) -> Optional[LaneGraphData]:
        scenario_index = int(scenario_index)
        if scenario_index in self._cache_by_record_index:
            return self._cache_by_record_index[scenario_index]

        self._build_index()
        assert self._zf is not None

        name = self._entry_by_record_index.get(scenario_index)
        if name is None:
            return None

        lane_graph = LaneGraphZipStore._load_lane_graph_from_npz_bytes(self._zf.read(name))
        self._cache_by_record_index[scenario_index] = lane_graph
        return lane_graph

class LaneGraphLoader:
    """Main interface for loading lane graphs for scenarios, with optional zip caching."""

    def __init__(self, lane_graph_dir: Optional[str] = None):
        self.lane_graph_dir = lane_graph_dir
        self._zip_stores = None

    def get_lane_graph_for_scenario(self, tfrecord_path: str, scenario_id: int) -> Optional[LaneGraphData]:
        if self.lane_graph_dir is None or tfrecord_path is None:
            return None

        zip_path = _lane_graph_zip_path_for_tfrecord(tfrecord_path, self.lane_graph_dir)
        if self._zip_stores is None or self._zip_stores[0] != zip_path.as_posix():
            self._zip_stores = (zip_path.as_posix(), LaneGraphShardStore(zip_path.as_posix()))
        
        return self._zip_stores[1].get_by_record_index(int(scenario_id))

    def get_lane_graph_for_scenarios(self, tfrecord_path: str, scenario_indices: list[int]) -> list[Optional[LaneGraphData]]:
        return [self.get_lane_graph_for_scenario(tfrecord_path, idx) for idx in scenario_indices]
    

def lane_start_pose_from_lane_graph(lane_graph, lane_id: int) -> np.ndarray | None:
    """Returns [x, y, heading] of a lane's start point from lane graph."""
    lane_range = lane_graph.lane_id_to_node_range.get(int(lane_id))
    if lane_range is None:
        return None
    start, end = int(lane_range[0]), int(lane_range[1])
    nodes_xyz = np.asarray(lane_graph.nodes_xyz, dtype=np.float32)
    if start < 0 or end > int(nodes_xyz.shape[0]):
        return None

    start_xy = nodes_xyz[start, :2]
    if end - start >= 2:
        next_xy = nodes_xyz[start + 1, :2]
        vec = next_xy - start_xy
    else:
        vec = np.asarray([1.0, 0.0], dtype=np.float32)

    if float(np.linalg.norm(vec)) <= 1e-6:
        heading = 0.0
    else:
        heading = float(math.atan2(float(vec[1]), float(vec[0])))
    return np.asarray([float(start_xy[0]), float(start_xy[1]), heading], dtype=np.float32)