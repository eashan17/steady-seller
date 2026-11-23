"""Synthesise the demo video's music bed (numpy only): 120 BPM, A minor (Am-F-C-G), mixed low to sit under a voiceover.
Scene cuts in the video fall on beats; soft swells lead into each cut.

    python demo/music.py out.wav [seconds] [cut,cut,...]
"""
import sys
import wave

import numpy as np

SR, BPM = 48000, 120
BEAT = 60 / BPM
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
CUTS = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else "14,34,94,126,146,170".split(","))]
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)

def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)):          # one-pole; fine for short note buffers
        acc = (1 - a) * x[i] + a * acc; y[i] = acc
    return y

def env(n, a, r):
    e = np.ones(n); na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, max(1, na)); e[-nr:] *= np.linspace(1, 0, max(1, nr))
    return e

CHORDS = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]   # Am F C G (root, voicing)
BAR = 4 * BEAT

pad = np.zeros(N); bass = np.zeros(N); arp = np.zeros(N); kick = np.zeros(N); hat = np.zeros(N); swell = np.zeros(N)

# pad: detuned saws, softened
nb = int(BAR * SR)
for b in range(int(DUR / BAR) + 1):
    root, voicing = CHORDS[b % 4]
    s0 = b * nb
    if s0 >= N: break
    n = min(nb, N - s0); tt = np.arange(n) / SR
    x = np.zeros(n)
    for m in voicing + [voicing[0] + 12]:
        for det in (-0.12, 0.12):
            f = hz(m) * 2 ** (det / 12)
            x += 2 * ((tt * f) % 1.0) - 1
    x = lowpass(x / 8, 900) * env(n, 0.35, 0.35)
    pad[s0:s0 + n] += x
    # bass: root eighths, plucky
    for k in range(8):
        e0 = s0 + int(k * BEAT / 2 * SR); ln = int(BEAT / 2 * SR * 0.9)
        if e0 + ln > N: break
        tb = np.arange(ln) / SR
        bass[e0:e0 + ln] += np.sin(2 * np.pi * hz(root - 12) * tb) * np.exp(-tb * 7) * (1.0 if k % 2 == 0 else 0.7)
    # arpeggio: 16ths over the voicing (used in the middle sections)
    for k in range(16):
        e0 = s0 + int(k * BEAT / 4 * SR); ln = int(BEAT / 4 * SR * 0.85)
        if e0 + ln > N: break
        m = (voicing + [voicing[0] + 12])[k % 4] + 12
        ta = np.arange(ln) / SR
        arp[e0:e0 + ln] += np.sin(2 * np.pi * hz(m) * ta) * np.exp(-ta * 14)

for k in range(int(DUR / BEAT)):
    s0 = int(k * BEAT * SR)
    if k % 2 == 0:                                   # kick on beats 1 and 3
        ln = int(0.25 * SR); tk = np.arange(ln) / SR
        f = 120 * np.exp(-tk * 18) + 45
        kick[s0:s0 + ln] += np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tk * 9)
    o = s0 + int(BEAT / 2 * SR); ln = int(0.05 * SR)  # hat on the off-beat
    if o + ln < N:
        hat[o:o + ln] += rng.standard_normal(ln) * np.exp(-np.arange(ln) / SR * 80)
hat = hat - lowpass(hat, 6000)

for c in CUTS:                                      # filtered-noise swell into each cut (2 beats long)
    s1 = int(c * SR); s0 = s1 - int(2 * BEAT * SR)
    if s0 < 0: continue
    ln = s1 - s0
    w = rng.standard_normal(ln) * np.linspace(0, 1, ln) ** 2
    swell[s0:s1] += lowpass(w, 2500)

def section(lo, hi, fade=1.0):
    g = np.zeros(N)
    a, b = int(lo * SR), int(min(hi, DUR) * SR)
    g[a:b] = 1.0
    k = int(fade * SR)
    g[a:a + k] *= np.linspace(0, 1, k) if a > 0 else 1.0
    return g

groove = np.clip(section(14, 170, 1.0) + 0.0, 0, 1)
arp_on = np.clip(section(34, 94, 2.0) * 0.6 + section(94, 126, 1.0) * 1.0 + section(146, 170, 1.0) * 0.6, 0, 1)
master_env = np.ones(N); master_env[:int(1.5 * SR)] = np.linspace(0, 1, int(1.5 * SR))
master_env[-int(4 * SR):] = np.linspace(1, 0, int(4 * SR))

mix = (0.55 * pad + 0.42 * bass * (0.35 + 0.65 * groove) + 0.10 * arp * arp_on + 0.55 * kick * groove
       + 0.10 * hat * (0.4 + 0.6 * groove) + 0.10 * swell) * master_env
# gentle stereo: pad/arp widened by a small delay
left = mix + 0.08 * np.roll(0.55 * pad + 0.1 * arp * arp_on, int(0.011 * SR))
right = mix + 0.08 * np.roll(0.55 * pad + 0.1 * arp * arp_on, int(0.017 * SR))
st = np.stack([left, right], 1)
st = st / np.max(np.abs(st)) * 0.5                 # peak -6 dBFS; the voice sits on top in the final mix
with wave.open(sys.argv[1] if len(sys.argv) > 1 else "music.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
print("wrote", DUR, "s")
