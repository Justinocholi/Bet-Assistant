"""Ensure the repository root is importable during tests.

CI runs plain ``pytest`` (prepend import mode), which does not put the repo root
on ``sys.path``. Tests that import root-level modules (e.g. ``app``) would then
fail to collect. Pytest discovers this root conftest and we add the repo root
explicitly so those imports resolve under any invocation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
