# Copyright 2025 Valeo.


"""Base observation wrapper for the agent."""

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import env as waymax_env

from vmax.simulator.features import extractor
from vmax.simulator.wrappers.base import Wrapper


class ObservationWrapper(Wrapper):
    """Observation wrapper that applies a feature extractor to the simulator state."""

    def __init__(
        self,
        env: waymax_env.PlanningAgentEnvironment,
        obs_type: str,
        obs_past_num_steps: int = 1,
        objects: dict | None = None,
        roadgraphs: dict | None = None,
        traffic_lights: dict | None = None,
        path_target: dict | None = None,
        goal: dict | None = None,
    ) -> None:
        """Initialize the base observation wrapper.

        Args:
            env: Environment to wrap.
            obs_type: Type of observation to extract.
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration for objects.
            roadgraphs_config: Configuration for roadgraphs.
            traffic_lights_config: Configuration for traffic lights.
            path_target_config: Configuration for the path target.
            goal: Configuration for the goal features.

        """
        super().__init__(env)
        self._features_extractor = extractor.get_extractor(obs_type)(
            obs_past_num_steps=obs_past_num_steps,
            objects_config=objects,
            roadgraphs_config=roadgraphs,
            traffic_lights_config=traffic_lights,
            path_target_config=path_target,
            goal_config=goal,
        )

        if obs_type in ["idm", "pdm"]:
            self._flatten_observation = False
        else:
            self._flatten_observation = True

    @property
    def features_extractor(self) -> extractor.AbstractFeaturesExtractor:
        """Return the features extractor."""
        return self._features_extractor

    def observation_spec(self) -> int:
        """Return the observation space size.

        This method calculates the size of the flattened observation vector
        that will be returned for a given scenario state.

        Args:
            scenario: The current simulator state.

        Returns:
            The observation space size (integer dimension of the observation vector).
        """
        # Calculate the size analytically instead of empirically if possible
        extractor = self.features_extractor

        # Get sizes from the extractor
        object_features_size = extractor.get_features_size(extractor._object_features_key)
        roadgraph_features_size = extractor.get_features_size(extractor._roadgraph_features_key)
        traffic_lights_features_size = extractor.get_features_size(extractor._traffic_lights_features_key)
        path_target_feature_size = extractor.get_features_size(extractor._path_target_features_key)

        # Calculate total size
        sdc_size = 1 * extractor._obs_past_num_steps * object_features_size
        other_objects_size = extractor._num_closest_objects * extractor._obs_past_num_steps * object_features_size
        roadgraph_size = extractor._roadgraph_top_k * roadgraph_features_size
        traffic_lights_size = (
            extractor._num_closest_traffic_lights * extractor._obs_past_num_steps * traffic_lights_features_size
        )
        path_target_size = extractor._num_target_path_points * path_target_feature_size
        goal_size = extractor.goal_features_size

        # Return total size
        return sdc_size + other_objects_size + roadgraph_size + traffic_lights_size + path_target_size + goal_size

    def observe(self, simulator_state: datatypes.SimulatorState, key: jax.Array = None) -> jax.Array:
        """Generate observation vector from simulator state.

        This method extracts features from the simulator state and concatenates them
        into a single flat vector for use by agents. The features include:
        - SDC trajectory features
        - Other agents' trajectory features
        - Roadgraph features
        - Traffic light features
        - GPS path features
        - Goal features (when enabled)

        Args:
            simulator_state: Current simulator state.
            key: Random key for observation noise. If None, no noise will be added.

        Returns:
            A flattened array containing all observation features concatenated.

        """
        features = self._features_extractor.extract_features(simulator_state, key)

        if not self._flatten_observation:
            return features

        # Reshape every feature block by batch dimension, then concatenate. Iterating
        # keeps this agnostic to how many blocks the extractor emits (the vec extractor
        # appends a goal block that the gt/segment extractors do not).
        matrices = [jnp.reshape(matrix, (*simulator_state.batch_dims, -1)) for matrix in features]

        return jnp.concatenate(matrices, axis=len(simulator_state.batch_dims))
