import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from viz.render import _load_scenario_state_fast, render_videos_batched  # noqa: E402
from viz import viz as viz_module  # noqa: E402

def _save_population_xy_score_image(
    *,
    sim_state,
    timestep: int,
    current_world_bkt5: np.ndarray,
    current_scores_bk: np.ndarray,
    output_path: Path,
    scenario_indices: list[int],
    front_x: float,
    back_x: float,
    front_y: float,
    back_y: float,
    target_lanes: list[np.ndarray] | None = None,
    target_vehicles: list[int] | None = None,
) -> None:
    
    world_bkt5 = np.asarray(current_world_bkt5, dtype=np.float32)
    scores_bk = np.asarray(current_scores_bk, dtype=np.float32)
    if world_bkt5.ndim != 4:
        raise ValueError(f"current_world_bkt5 must be rank-4 [B, K, T, 5], got {world_bkt5.shape}")
    if scores_bk.ndim != 2:
        raise ValueError(f"current_scores_bk must be rank-2 [B, K], got {scores_bk.shape}")

    batch_size = int(world_bkt5.shape[0])
    if int(scores_bk.shape[0]) != batch_size or int(scores_bk.shape[1]) != int(world_bkt5.shape[1]):
        raise ValueError(
            "current_world_bkt5 and current_scores_bk shapes are inconsistent: "
            f"{world_bkt5.shape} vs {scores_bk.shape}"
        )

    ncols = batch_size
    cmap = plt.get_cmap("viridis")

    for b in range(batch_size):
        output_path_scenario = output_path / f"scenario_{scenario_indices[b]:05d}"
        output_path_scenario.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 1, figsize=(6.0, 6.0), squeeze=False)
        ax = axes[0, 0]
        state_b = viz_module._index_pytree(sim_state, b)
        t = int(np.clip(timestep, 0, int(np.asarray(state_b.log_trajectory.x).shape[-1]) - 1))

        viz_module.plot_roadgraph_points(ax, state_b.roadgraph_points, verbose=False)

        if target_lanes is not None and b < len(target_lanes):
            if target_lanes[b] is not None:
                lane_points = np.asarray(target_lanes[b], dtype=np.float32)
                if lane_points.ndim == 2 and lane_points.shape[0] >= 2 and lane_points.shape[1] >= 2:
                    ax.plot(
                        lane_points[:, 0],
                        lane_points[:, 1],
                        color="deepskyblue",
                        linewidth=3.0,
                        alpha=0.95,
                        zorder=5,
                    )

        is_ego = np.asarray(state_b.object_metadata.is_sdc).astype(bool)
        num_obj = int(state_b.log_trajectory.num_objects)
        is_controlled = np.zeros((num_obj,), dtype=bool)
        is_adv = np.zeros((num_obj,), dtype=bool)
        if target_vehicles is not None and b < len(target_vehicles):
            target_idx = target_vehicles[b]
            if target_idx is not None:
                target_idx = int(target_idx)
                if 0 <= target_idx < num_obj:
                    is_adv[target_idx] = True
        viz_module.plot_trajectory(
            ax,
            state_b.log_trajectory,
            is_controlled=is_controlled,
            time_idx=t,
            indices=None,
            past_traj_length=0,
            is_ego=is_ego,
            is_adv=is_adv,
        )

        traj_kt2 = world_bkt5[b, :, :, :2]
        score_k = scores_bk[b]
        for k in range(int(traj_kt2.shape[0])):
            xy_t2 = traj_kt2[k]
            ax.plot(
                xy_t2[:, 0],
                xy_t2[:, 1],
                color=cmap(float(score_k[k])),
                linewidth=1.2,
                alpha=0.95,
            )
            ax.scatter(
                xy_t2[0, 0],
                xy_t2[0, 1],
                color=cmap(float(score_k[k])),
                s=8,
                alpha=0.9,
            )

        current_xy = np.asarray(state_b.log_trajectory.xy[:, t, :])
        if np.any(is_ego):
            center_xy = current_xy[is_ego][0]
        else:
            center_xy = np.nanmean(current_xy, axis=0)
        ax.axis(
            (
                float(center_xy[0]) - float(back_x),
                float(center_xy[0]) + float(front_x),
                float(center_xy[1]) - float(back_y),
                float(center_xy[1]) + float(front_y),
            )
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"batch={b}, t={t}, K={traj_kt2.shape[0]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.2)

        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="score")
        fig.tight_layout()

        fig.savefig(output_path_scenario / f"step_{t:04d}.png", dpi=160)
        plt.close(fig)