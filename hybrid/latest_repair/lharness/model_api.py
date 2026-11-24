"""Connect the coding agent to your own vision-language model.

RoboEvo2 ships no model client. Implement ``query_model`` below with the API
you use. Everything else (prompts, images, skill catalog, memory, budgets and
parsing of the reply) is handled by the harness.

The same contract is used by the Hybrid evaluator (``hybrid/full_campaign``),
the targeted repair runner (``hybrid/latest_repair``, which keeps its own copy
of this file) and the evolution planner (``evolution/libero_runtime.py``).

Input
-----
``messages`` is a list of two messages::

    [{"role": "system", "content": "<instructions, a string>"},
     {"role": "user", "content": [<block>, ...]}]

Each user block is one of::

    {"type": "text", "text": "<string>"}
    {"type": "image_url", "image_url": {"url": "/absolute/path/to/image.png"}}

The first text block is a JSON object with the instruction, camera calibration,
robot state, remaining budget, skill catalog and retrieved memory. Each image
is a PNG on the local disk, preceded by a text block naming its camera and
view (RGB or a depth preview). Send the images in the given order; most APIs
take them as base64 (see ``image_base64``).

``settings`` is the ``coding_agent`` block of your config (empty for the
evolution planner), so you can add your own keys there, for example a model
name or a timeout. When the caller has a deadline it also contains
``case_deadline_monotonic``, a ``time.monotonic()`` value; stop waiting by then.

Output
------
Return the model's reply as a string. The instructions ask for exactly one JSON
object; enable your API's JSON output mode if it has one. Raise an exception
when the request fails. The harness records it as an infrastructure error, not
as a task failure, so do not return placeholder text.

Keep credentials out of the repository, for example in environment variables.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path


def image_base64(block):
    """Return the PNG bytes of an image block, base64-encoded."""
    return base64.b64encode(Path(block["image_url"]["url"]).read_bytes()).decode("ascii")


def query_model(messages, settings):
    """Send ``messages`` to your model and return its reply text."""
    raise NotImplementedError(
        "Connect a model: implement query_model() in lharness/model_api.py "
        "(see the module docstring and 'Connect your model' in the README).")


class CodingAgent:
    """Records each request and its reply next to the case outputs."""

    def __init__(self, settings):
        self.settings = dict(settings)

    def query(self, messages, *, json_mode=True):
        if self.settings.get("enabled") is False:
            raise RuntimeError("The coding agent is disabled in the config")
        directory = Path(self.settings["io_dir"]).resolve() / "api"
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "messages.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        settings = {k: v for k, v in self.settings.items() if k != "io_dir"}
        content = query_model(messages, settings)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("query_model() must return the reply text")
        (directory / "response.txt").write_text(content)
        return {"content": content}
