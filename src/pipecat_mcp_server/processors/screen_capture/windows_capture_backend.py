#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Stub screen capture backend for Windows.

Screen capture is not currently implemented on Windows. This backend provides
noop implementations so the pipeline can still be constructed without errors.
"""

from typing import List, Optional, Tuple

from .base_capture_backend import BaseCaptureBackend, WindowInfo


class WindowsCaptureBackend(BaseCaptureBackend):
    """Stub backend for Windows — screen capture is not yet implemented."""

    async def list_windows(self) -> List[WindowInfo]:
        return []

    async def start(self, window_id: Optional[int], monitor: int) -> Optional[int]:
        return None

    async def capture(self) -> Optional[Tuple[bytes, Tuple[int, int]]]:
        return None

    async def stop(self) -> None:
        pass
