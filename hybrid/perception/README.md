# SAM3 perception service

The Hybrid repair runner can request drawer and tabletop masks from this local service. It sends a current RGB image and a short text prompt to `POST /segment`. The service applies the requested score threshold before encoding binary masks. The repair code combines the proposals with current RGB-D, calibration, and measured robot state to inform its next plan.

Use a Python 3.12 environment with Meta SAM3, a CUDA-compatible PyTorch build, NumPy, Pillow, FastAPI, Pydantic, Uvicorn, and Tyro. Supply a local SAM3 checkpoint.

```sh
python sam3_filtered_server.py --host 127.0.0.1 --port PORT \
  --device cuda:0 --checkpoint "$SAM3_CHECKPOINT"
```

In the repair configuration, enable `coding_agent.public_visual_proposals` and set its `endpoint` to `http://127.0.0.1:PORT/segment`. Set `tabletop_enabled` within the proposal config to use tabletop masks. For concurrent workers, set `admission_root` to a shared, writable local directory.

The server derives from [RATs at commit 1fd2ef5](https://github.com/Playful-RATs/RATs/tree/1fd2ef56857cfde6d5d815fc49a9fe9b0a531beb). The included Apache 2.0 license covers that server source. Obtain SAM3 and its weights under their own terms.
