"""Fixed pipeline settings. Chosen once and cited, never searched over."""

import hashlib
import json
from dataclasses import asdict, dataclass, field

FS = 360  # MIT-BIH sampling frequency, Hz


@dataclass(frozen=True)
class FilterConfig:
    """Baseline removal and low-pass settings.

    Baseline: de Chazal et al. (2004) cascade of 200 ms and 600 ms median filters.
    Low-pass: zero-phase 4th-order Butterworth at 35 Hz (de Chazal used a 12-tap FIR at
    35 Hz; the IIR+filtfilt version is a documented deviation with no phase distortion).
    """

    baseline_windows_s: tuple[float, float] = (0.2, 0.6)
    lowpass_hz: float = 35.0
    lowpass_order: int = 4


@dataclass(frozen=True)
class WindowConfig:
    """Fixed window around the annotated R peak.

    0.25 s before covers P-wave onset (PR <= 0.20 s); 0.45 s after covers T-wave end at
    normal rates (QT <~ 0.44 s). At 360 Hz this is 90 + 162 = 252 samples.
    """

    before_s: float = 0.25
    after_s: float = 0.45

    def samples(self, fs: float) -> tuple[int, int]:
        """Return (samples before, samples after) the R peak at sampling rate fs."""
        return round(self.before_s * fs), round(self.after_s * fs)


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that determines the content of a cached beat set."""

    filter: FilterConfig = field(default_factory=FilterConfig)
    window: WindowConfig = field(default_factory=WindowConfig)

    def digest(self, record_ids: frozenset[int]) -> str:
        """Stable short hash of this config plus the record IDs, used as a cache key."""
        payload = json.dumps({"cfg": asdict(self), "records": sorted(record_ids)})
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ImageConfig:
    """2D input: each beat window drawn as a line plot, per-beat min-max scaled."""

    size: int = 224
    line_width: int = 2
