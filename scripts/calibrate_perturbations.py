"""Measures what each perturbation strength does, by replaying recorded baseline actions.

Physics replay is exact and needs no rendering or policy, so hundreds of trials take minutes. The
recorded actions do not react to the perturbation; this matches the real executor until its next
re-query (at most 100 steps). Uses baseline seeds only, never calib or test.

Usage: python scripts/calibrate_perturbations.py <baseline_run_dir> <out_csv>
"""

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from amon.settings import Settings, TaskSettings, load_settings
from amon.sim.env import AlohaSim
from amon.sim.labels import find_drops
from amon.sim.perturb import TARGET, make_perturbation
from amon.sim.state import StateReader

HORIZON = 60  # steps after firing over which the motion is described
BOUNCE_HEIGHT_M = 0.005  # a bounce is a rise this far above rest after landing
KNOCK_GRID = [
    {"force_n": force, "steps": steps, "direction": direction}
    for direction, steps, force in itertools.product(
        ["random", "across_fingers"], [5, 10], [3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
    )
]
GRIDS = {
    ("transfer_cube", "knock"): KNOCK_GRID,
    ("insertion", "knock"): KNOCK_GRID,
    ("transfer_cube", "slip"): [
        {"friction_scale": scale, "steps": steps}
        for scale, steps in itertools.product([0.01, 0.1, 0.3], [5, 10, 20, 40])
    ],
    ("transfer_cube", "nudge"): [
        {"offset_m": offset, "trigger_radius_m": radius}
        for offset, radius in itertools.product([0.02, 0.03, 0.05], [0.08, 0.10, 0.12])
    ],
    ("insertion", "peg_misalign"): [{"angle_deg": (5.0, 10.0)}, {"angle_deg": (10.0, 20.0)}],
}


def run_trial(
    sim: AlohaSim, task: TaskSettings, name: str, settings: Settings, trace: Path
) -> dict[str, float | bool | int]:
    """Replays one recorded episode with the perturbation and describes what happened."""
    with np.load(trace) as arrays:
        actions, seed = arrays["action_exec"], int(arrays["seed"])
    sim.reset(seed)
    reader = StateReader(sim.physics, task, settings)
    perturbation = make_perturbation(name, task, settings, seed)
    rest = next(iter(task.objects.values())).rest_height_m
    phase = 0
    held: list[np.ndarray] = []
    position: list[np.ndarray] = []
    success = False
    for step, action in enumerate(actions):
        state = reader.read(sim.physics)
        held.append(state.held)
        position.append(state.obj_pos[TARGET])
        perturbation.before_step(sim.physics, step, state, phase)
        hook = perturbation.substep if perturbation.substep_hook_needed() else None
        reward = sim.step_without_rendering(action, hook)
        phase = max(phase, reward)
        success = success or reward == task.success_reward
        if success:
            break
    fired = perturbation.step_fired
    perturbation.restore(sim.physics)
    if fired < 0:
        return {"fired": False}
    path = np.array(position)[fired : fired + HORIZON]
    drops = [
        event.onset
        for event in find_drops(np.array(held), settings.labels.drop_debounce_steps)
        if fired <= event.onset < fired + HORIZON
    ]
    speed = np.linalg.norm(np.diff(path, axis=0), axis=1) * settings.control_hz
    height = path[:, 2] - rest
    landed = np.flatnonzero(height < BOUNCE_HEIGHT_M)
    bounces = 0
    if len(landed) and name != "nudge":
        airborne = height[landed[0] :] > BOUNCE_HEIGHT_M
        bounces = int(np.sum(~airborne[:-1] & airborne[1:]))
    return {
        "fired": True,
        "success": success,
        "dropped": bool(drops),
        "steps_to_drop": drops[0] - fired if drops else -1,
        "peak_speed": float(speed.max()) if len(speed) else 0.0,
        "peak_rise": float(height.max() - height[0]),
        "travel_after_drop": float(np.linalg.norm(path[-1, :2] - path[0, :2])) if drops else 0.0,
        "bounces": bounces,
        "off_table": bool(path[:, 2].min() < rest - 0.01),
        "fire_height": float(height[0]),
    }


def main() -> None:
    """Runs every grid and writes one summary row per setting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--task", help="only this task")
    parser.add_argument("--only", nargs="*", help="perturbation names")
    arguments = parser.parse_args()
    base = load_settings()
    rows = []
    for (task_name, name), grid in GRIDS.items():
        if arguments.task not in (None, task_name):
            continue
        if arguments.only and name not in arguments.only:
            continue
        task = base.tasks[task_name]
        sim = AlohaSim(task)
        traces = sorted((arguments.baseline / "baseline" / task_name).glob("ep_*.npz"))
        # Nominal successes only, so that every failure is caused by the perturbation.
        traces = [path for path in traces if bool(np.load(path)["success"])][: arguments.trials]
        for values in grid:
            tuned = replace(base.perturbations[name], **values)
            settings = replace(base, perturbations={**base.perturbations, name: tuned})
            trials = pd.DataFrame([run_trial(sim, task, name, settings, path) for path in traces])
            fired = trials[trials["fired"]].infer_objects()
            rows.append(
                {
                    "task": task_name,
                    "perturbation": name,
                    **{key: str(value) for key, value in values.items()},
                    "trials": len(trials),
                    "fired": len(fired),
                    "dropped": float(fired["dropped"].mean()),
                    "success": float(fired["success"].mean()),
                    "steps_to_drop": float(fired.loc[fired["dropped"], "steps_to_drop"].median()),
                    "speed_p50": float(fired["peak_speed"].median()),
                    "speed_max": float(fired["peak_speed"].max()),
                    "rise_max": float(fired["peak_rise"].max()),
                    "travel_p50": float(fired.loc[fired["dropped"], "travel_after_drop"].median()),
                    "travel_max": float(fired["travel_after_drop"].max()),
                    "bounces_mean": float(fired["bounces"].mean()),
                    "bounces_gt1": float((fired["bounces"] > 1).mean()),
                    "off_table": float(fired["off_table"].mean()),
                }
            )
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()})
        sim.close()
    pd.DataFrame(rows).to_csv(arguments.out, index=False)


if __name__ == "__main__":
    main()
