# Copyright 2025 Valeo.

"""Constants for the simulator."""

MAX_ACCEL_BICYCLE = 6.0
MAX_STEERING = 0.3
TIME_DELTA = 0.1

NUM_SDC_PATHS = 10
NUM_POINTS_PER_SDC_PATH = 300

# Raw WOMD 1.3.1 tf_example ships its own curated SDC routes, but at different
# dimensions than ScenarioMax re-emits them at. Reading a WOMD shard with the
# ScenarioMax dims above fails in the Waymax dataloader's reshape.
WOMD_NUM_SDC_PATHS = 45
WOMD_NUM_POINTS_PER_SDC_PATH = 800
