"""Print duration / level / transient positions for every SFX clip (run inside Blender for its mp3 decoder).
blender -b --factory-startup -P sfx_probe.py
"""
from pathlib import Path
import numpy as np
import aud

SFX = Path(__file__).resolve().parent.parent / "sfx"

for p in sorted(SFX.glob("*.mp3")):
    s = aud.Sound(str(p))
    rate = int(s.specs[0])
    x = np.asarray(s.data(), dtype=np.float32)
    mono = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    dur = len(mono) / rate
    peak = float(mono.max())
    rms = float(np.sqrt(np.mean((x.astype(np.float64)) ** 2)))
    # 10 ms envelope, onsets = jumps of > 4x over the previous 30 ms
    hop = rate // 100
    env = np.array([mono[i:i + hop].max() for i in range(0, len(mono) - hop, hop)])
    ons = []
    for i in range(3, len(env)):
        prev = env[i - 3:i].mean() + 1e-4
        if env[i] > 0.12 * peak and env[i] > 4 * prev and (not ons or i - ons[-1] > 8):
            ons.append(i)
    loud = np.where(env > 0.1 * peak)[0]
    span = (loud[0] / 100, loud[-1] / 100) if len(loud) else (0, 0)
    shape = "".join(" .:-=+*#%@"[min(9, int(9 * v / (peak + 1e-9)))] for v in env[:: max(1, len(env) // 60)])
    print(f"{p.stem:9s} {dur:5.2f}s ch={x.shape[1] if x.ndim == 2 else 1} rate={rate} peak={peak:.2f} "
          f"rms={rms:.3f} loud={span[0]:.2f}-{span[1]:.2f}s onsets={[round(o / 100, 2) for o in ons][:12]}")
    print(f"          |{shape}|")
