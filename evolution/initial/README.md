# Round zero

`state.json` is a portable LIBERO starting harness for a fresh evolution run. Its five skills use trusted public control primitives, its working and episodic memory start empty, and its three routing instructions govern access to demonstrations and the current policy. Supply the initial policy trained on robot demonstrations and the authorized training demonstrations separately. Each call to `initial_state()` returns an independent copy.

The public observation, gripper and motion interfaces follow the reduced Franka APIs from CaP-X and RATs. The LIBERO controller grounds a pixel in the current depth image using camera calibration. A deployment exposes only the public controller to the dispatcher. Object poses, contacts and native success are evaluator-private. [CaP-X Franka API](https://github.com/capgym/cap-x/blob/53e9966d7a8e2fa7494676772bccc35280f5c0ed/capx/integrations/franka/libero_reduced.py) · [RATs Franka API](https://github.com/Playful-RATs/RATs/blob/1fd2ef56857cfde6d5d815fc49a9fe9b0a531beb/rats/integrations/franka/molmospaces_reduced.py)

The inspect, grasp, place and relative-clearance programs follow the recipe order and phase budgets of the LIBERO Hybrid harness in [`hybrid/full_campaign/skills/catalog.json`](../../hybrid/full_campaign/skills/catalog.json). The fifth skill, bounded policy continuation, is the instruction-conditioned Code+Policy call.

`compile_skill` returns a program in the controller's bounded operation grammar. `execute_skill` executes that program using the trusted controller's action budget, camera grounding, and external native stop. Its arguments carry relative offsets in metres, axis-angle increments in radians, image pixels in horizontal/vertical order, and gripper commands from −1 (open) to +1 (closed). A caller selects a visible target anew for each invocation. To inspect a program on CPU from the repository root, run

```sh
python - <<'PY'
from evolution.initial.skills import compile_skill, initial_state

state = initial_state()
ops = compile_skill(state, "code", "relative_clearance", {
    "delta": [0, 0, 0.04], "gripper": 1,
})
print(ops)
PY
```
