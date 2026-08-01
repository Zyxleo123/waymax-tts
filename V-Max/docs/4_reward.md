# V-Max Reward Function Documentation

## Overview

In V-Max, the reward function is a key component that guides the behavior of autonomous agents during simulation. Rewards are used to evaluate how well an agent is performing with respect to safety, comfort, rule compliance, and task completion. By designing appropriate reward functions, you can encourage agents to drive safely, follow traffic rules, and achieve their objectives efficiently.

## How Rewards Work in V-Max

V-Max uses a flexible reward system based on wrappers. The main reward wrapper is `RewardLinearWrapper`, which computes the total reward as a weighted sum of several individual reward functions. Each function measures a specific aspect of driving behavior (e.g., staying on the road, avoiding collisions, obeying speed limits).

You can configure which reward functions to use and their relative importance by providing a `reward_config` dictionary, where keys are reward function names and values are their weights.

## Configuring Rewards

To use the reward system, wrap your environment with `RewardLinearWrapper` and provide a configuration, for example:

```python
reward_config = {
    "overlap": -10.0,           # Penalize collisions
    "offroad": -5.0,            # Penalize driving off the road
    "progression": 1.0,         # Reward making progress
    "comfort": 0.5,             # Reward smooth driving
    # ... add more as needed
}

env = RewardLinearWrapper(env, reward_config)
```

Each time the agent takes an action, the wrapper computes the total reward as:

```
reward = sum(weight * reward_fn(state) for each reward_fn in config)
```

## Available Reward Functions

Here are the main reward functions you can use in your configuration:

| Name                | Description                                                      |
|---------------------|------------------------------------------------------------------|
| `overlap`           | Penalizes collisions with other objects                           |
| `offroad`           | Penalizes driving off the road                                   |
| `off_route`         | Penalizes deviating from the planned route                       |
| `below_ttc`         | Penalizes unsafe time-to-collision situations                    |
| `red_light`         | Penalizes running red lights                                     |
| `overspeed`         | Penalizes exceeding the speed limit                              |
| `driving_direction` | Penalizes driving in the wrong direction                         |
| `lane_deviation`    | Penalizes deviating from the intended lane                       |
| `log_div_clip`      | Penalizes large deviations from the expected trajectory           |
| `log_div`           | Measures log divergence from the expected trajectory              |
| `progression`       | Rewards making forward progress along the route                  |
| `comfort`           | Rewards smooth and comfortable driving                           |
| `reached_goal`      | Per-step bonus while within the goal radius (reach *and stay*)    |
| `reached_goal_once` | One-time bonus on the first step the goal is reached              |
| `goal_progress`     | Dense reward for closing distance to the goal (driving *towards* it) |

### Goal-reaching rewards

The goal is the SDC's destination, defined as its **last valid logged position**
(the same definition used to evaluate goal reaching elsewhere in the project).
Two variants are provided:

- `reached_goal` (dense): returns `True` on every step the SDC is within
  `goal_threshold` meters (default 3.0 m) of the goal. Rewards reaching *and*
  remaining at the destination, but can be "farmed" by camping near the goal, so
  keep its weight modest.
- `reached_goal_once` (sparse): returns `True` only on the **first** step the SDC
  enters the goal radius, and never again in the episode. Implemented statelessly
  by inspecting the simulated trajectory history in the state (no external
  `goal_reached` flag needed), and robust to leaving/re-entering the radius. Best
  suited to a single, larger goal bonus.

Use one or the other (using both double-counts reaching the goal).

Both of the above are purely **proximity-based**: they only look at the SDC's
*current* distance to the goal, not at whether it is actually moving towards
it. To reward driving *towards* the goal, use:

- `goal_progress` (dense): returns the reduction in distance-to-goal since the
  previous step (`previous_distance - current_distance`). Positive when the SDC
  moves closer to the goal, negative when it moves away, and roughly zero when
  moving perpendicular to the goal direction. This complements `reached_goal`/
  `reached_goal_once` by shaping behavior *throughout* the episode instead of
  only near the destination, and can be combined with either of them.

- **Penalty-based rewards** (e.g., `overlap`, `offroad`) usually return `True` (1.0) when a violation occurs, so you should assign them negative weights.
- **Reward-based rewards** (e.g., `progression`, `comfort`) return positive values when the agent behaves well, so you should assign them positive weights.

## Potential-Based Goal Shaping

Configured under the top-level `goal_shaping` key (not `reward_config`), and applied by
`BraxWrapper` rather than the reward wrapper, because it needs the termination flag:

```yaml
goal_shaping:
  coef: 1.0                     # 0.0 disables
  discount: ${algorithm.discount}
  normalize: true
```

It adds `F(s, s') = coef * (discount * Phi(s') - Phi(s))` with `Phi(s) = -distance_to_goal`
(divided by the initial distance when `normalize`).

Why this and not `goal_progress`: the discounted shaping telescopes over an episode to
`discount^T * Phi(s_T) - Phi(s_0)`, which depends only on where the episode started and
ended -- **never on the route taken between them**. That makes it provably policy-invariant
(Ng, Harada & Russell, 1999): it cannot be farmed by cutting a corner off-road, which is
exactly the failure mode a raw distance-closed reward like `goal_progress` has. So `coef`
can be raised to speed up credit assignment without distorting the safety trade-off, and
you do not need counter-penalties (route-following, lateral deviation) to suppress
beelining.

Two details are load-bearing:
- The `discount` factor is what makes the sum telescope. It **must** match
  `algorithm.discount`, or the policy-invariance guarantee is void.
- `Phi` is zeroed at absorbing states via the same `flag` that gates the critic's
  bootstrap. A time-limit truncation is *not* absorbing and correctly keeps `Phi(s')`.

Shaping densifies the sparse `reached_goal_once` bonus; it does not replace it. Keep both.
It is applied to the training env only, so evaluation reward stays equal to the true task
reward.

> Shaping fixes credit assignment, not observability. It only pays off if the goal is in
> the observation (`observation_config.goal`); a policy that cannot see the goal or the
> time remaining is being scored on state it cannot observe, and no shaping fixes that.

## Custom Rewards

If you need a custom reward function, you can subclass `RewardCustomWrapper` and implement your own logic in the `reward` method.

```python
class MyCustomRewardWrapper(RewardCustomWrapper):
    def reward(self, state, action):
        # Your custom reward logic here
        return jnp.array(...)
```

## Extending the Reward System

To add a new reward function:
1. Implement a new function (e.g., `_compute_my_new_reward(state)`) in `reward.py`.
2. Add it to the `_get_reward_fn` dictionary.
3. Use its name in your `reward_config`.

## Tips
- Tune the weights in `reward_config` to balance between safety, efficiency, and comfort.
- Use negative weights for penalties and positive weights for desirable behaviors.
- Test your configuration to ensure the agent learns the intended behavior.

## Summary
- The reward system in V-Max is modular and configurable.
- Use `RewardLinearWrapper` with a `reward_config` dictionary to combine multiple reward functions.
- Choose and tune reward functions to match your simulation goals.
- Extend or customize as needed for your use case.

For more details, see the source code in `vmax/simulator/wrappers/reward.py`.
