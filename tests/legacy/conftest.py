"""Fixtures for characterization tests of the frozen 2019 scripts/ code."""

import sys
import types

# The legacy cnn_model imports cv2 at module level but only uses it to read PNGs in
# getSignalDataFrame, which these tests never call. Stubbing it avoids adding OpenCV
# as a dependency for code that is deleted in Phase C.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
