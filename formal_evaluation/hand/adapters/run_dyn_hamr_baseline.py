#!/usr/bin/env python3
"""Dyn-HaMR canonical importer; see import_hand_motion_baseline.py for its input contract."""

from __future__ import annotations

import sys

from formal_evaluation.hand.adapters.import_hand_motion_baseline import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--method", "dyn_hamr"]
    main()
