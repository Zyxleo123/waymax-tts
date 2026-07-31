"""Block layout for the structured Waymax observation, aligned to V-Max.

The constants here mirror the `observation_config` of **`repro_sac_v2`**, the
V-Max SAC run that reached ~97% on ScenarioMaxWaymoValid
(`/zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2/.hydra/config.yaml`),
so the SB3 path sees the same information V-Max did -- but computed directly from
raw WOMD, with no ScenarioMax conversion.

Each block is a grid of ``count`` entities x ``steps`` timesteps; every row is
``[feat_0 .. feat_{d-1}, valid]`` (the trailing validity bit is omitted for
blocks whose entities are always present, e.g. ``path_target``). This mirrors
V-Max's ``VecFeaturesExtractor.unflatten_features``, which splits each block's
last channel off as the mask.

``rl/waymax_env.py`` writes this layout and ``rl/encoders.py`` reads it, so both
import from here rather than duplicating offsets.

V-Max reference (repro_sac_v2)::

    obs_past_num_steps: 5
    objects:        waypoints, velocity, yaw, size, valid   num_closest 16
    roadgraphs:     waypoints, direction, valid             element_types [road_edge]
                    top_k 200, interval 2, max_meters 70
                    meters_box {front 70, back 5, left 20, right 20}
    traffic_lights: waypoints, state, valid                 num_closest 5
    path_target:    waypoints                               num_points 3, points_gap 12
"""

from __future__ import annotations

import dataclasses

# --- V-Max observation_config constants (repro_sac_v2) --------------------- #
OBS_PAST_NUM_STEPS = 5      # objects + traffic lights carry this much history
NUM_CLOSEST_OBJECTS = 16    # observation_config.objects.num_closest_objects
ROADGRAPH_TOP_K = 200       # observation_config.roadgraphs.roadgraph_top_k
ROADGRAPH_INTERVAL = 2      # keep every Nth point before top-k
NUM_CLOSEST_TRAFFIC_LIGHTS = 5
PATH_TARGET_NUM_POINTS = 3  # observation_config.path_target.num_points
PATH_TARGET_POINTS_GAP = 12

# Normalisation. V-Max normalises positions/sizes by ``max_meters`` and clips.
MAX_METERS = 70.0           # observation_config.roadgraphs.max_meters
MAX_SPEED = 30.0            # V-Max features.MAX_SPEED (velocity normaliser)

# Ego-frame crop for roadgraph points (observation_config.roadgraphs.meters_box).
# Front-biased, not a radius: the policy needs the road ahead, not behind.
METERS_BOX_FRONT = 70.0
METERS_BOX_BACK = 5.0
METERS_BOX_LEFT = 20.0
METERS_BOX_RIGHT = 20.0

# element_types: [road_edge] -> the road_edge meta-type in V-Max's TYPE_MAP.
# waymax MapElementIds: ROAD_EDGE_BOUNDARY=15, ROAD_EDGE_MEDIAN=16.
# Keeping *only* road edges is deliberate: they are what the offroad metric is
# computed against, and lane centrelines would otherwise crowd them out of top-k.
ROADGRAPH_ELEMENT_TYPES: tuple[int, ...] = (15, 16)

# Traffic-light state one-hot width. V-Max's get_feature_size("state") returns
# max(TL_MAPPING) = max(range(9)) = 8, so its one-hot is 8 wide (not 9).
NUM_TL_STATES = 8


@dataclasses.dataclass(frozen=True)
class ObsBlock:
    """A ``count`` x ``steps`` grid of entity/timestep tokens.

    ``feat_dim`` counts real features only; when ``has_valid`` the row carries one
    extra trailing validity bit that the encoder splits off as its mask.
    """

    name: str
    count: int
    steps: int
    feat_dim: int
    has_valid: bool = True

    @property
    def stride(self) -> int:
        """Row width including the trailing valid bit."""
        return self.feat_dim + (1 if self.has_valid else 0)

    @property
    def num_tokens(self) -> int:
        """Attention tokens this block contributes (entities x timesteps)."""
        return self.count * self.steps

    @property
    def size(self) -> int:
        """Total flat width contributed by this block."""
        return self.num_tokens * self.stride


# Block order == concatenation order in ``waymax_env._compute_observation``.
# Do not reorder without updating that function.
#
# Per-object features (V-Max objects config): xy(2), vel_xy(2), yaw(1),
# length(1), width(1) = 7, plus valid.
# Roadgraph: xy(2), dir_xy(2) = 4, plus valid.
# Traffic lights: xy(2), state one-hot(8) = 10, plus valid.
# path_target: xy(2), no valid bit (V-Max masks it all-ones).
# goal: our own addition -- V-Max's repro_sac_v2 has no goal block, but this task
# is goal-reaching and the reward is defined against the goal, so the policy has
# to be able to see it.
DEFAULT_OBS_BLOCKS: tuple[ObsBlock, ...] = (
    ObsBlock("sdc", 1, OBS_PAST_NUM_STEPS, 7),
    ObsBlock("agents", NUM_CLOSEST_OBJECTS, OBS_PAST_NUM_STEPS, 7),
    ObsBlock("roadgraph", ROADGRAPH_TOP_K, 1, 4),
    ObsBlock("traffic_lights", NUM_CLOSEST_TRAFFIC_LIGHTS, OBS_PAST_NUM_STEPS, 10),
    ObsBlock("path_target", PATH_TARGET_NUM_POINTS, 1, 2, has_valid=False),
    ObsBlock("goal", 1, 1, 5),
)


def obs_layout_size(blocks: tuple[ObsBlock, ...] = DEFAULT_OBS_BLOCKS) -> int:
    """Total flat observation width for ``blocks``."""
    return sum(b.size for b in blocks)


def block_offsets(blocks: tuple[ObsBlock, ...] = DEFAULT_OBS_BLOCKS) -> list[int]:
    """Starting flat index of each block, in order."""
    offsets: list[int] = []
    running = 0
    for b in blocks:
        offsets.append(running)
        running += b.size
    return offsets


def num_tokens(blocks: tuple[ObsBlock, ...] = DEFAULT_OBS_BLOCKS) -> int:
    """Total attention tokens the LQ encoder will see."""
    return sum(b.num_tokens for b in blocks)
