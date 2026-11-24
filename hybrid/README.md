# LIBERO-PRO Hybrid

| Directory | Entry point | When to use it |
|---|---|---|
| `full_campaign/` | `run.py` | Start each episode at a benchmark initial state. The policy goes first; the harness can recover in the same episode. |
| `latest_repair/` | `run.py` | Revisit recorded policy failures. The runner checks saved actions against public RGB-D and robot feedback before entering recovery. |
| [`perception/`](perception/) | `sam3_filtered_server.py` | Serve optional SAM3 mask proposals from the current camera RGB. |

Each directory has its own `lharness/`, `skills/`, and `memory/`.

## Actor inputs and checks

The policy and planner receive the task instruction, configured RGB-D camera frames and calibration, and robot proprioception (joint, end-effector, and gripper measurements). The controller projects a fixed observation schema, and the planner rebuilds it before sending a request. The planner submits bounded motion primitives; it cannot call simulator APIs. Targets are selected from pixels in the current view and projected using aligned depth and calibration. Gripper readings and visual changes are evidence, not a grasp verdict. The planner reaches your model through `lharness/model_api.py` in each directory; see [Connect your model](../README.md#connect-your-model).

In `latest_repair/`, `lharness/public_extra_view.py` adds a fixed recovery camera. `lharness/visual_proposals.py` can request optional SAM proposals from the current RGB image. `lharness/public_support_geometry.py`, `lharness/public_orientation.py`, and the controller check visible geometry and robot motion. These checks provide feedback for the next plan. Only the evaluator uses the native success predicate, to stop at first success and score the episode. Object poses, contacts, attachment state, task IDs, rewards, and the evaluator verdict are not planner inputs.

Start with the [root setup](../README.md#setup-and-cpu-preflight) for a fresh cohort, or the [targeted repair setup](latest_repair/README.md) to replay saved prefixes.
