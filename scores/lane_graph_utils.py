from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class LaneGraphData:
    scenario_id: str
    lane_ids: np.ndarray
    nodes_xyz: np.ndarray
    node_lane_ids: np.ndarray
    node_point_indices: np.ndarray
    polyline_edges: np.ndarray
    successor_edges: np.ndarray
    predecessor_edges: np.ndarray
    left_neighbor_edges: np.ndarray
    right_neighbor_edges: np.ndarray
    lane_id_to_node_range: Dict[int, Tuple[int, int]]


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
                lane_ids=lane_ids,
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
