"""Estimate background outside glyphs, then erase glyphs before smoothing."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class BackgroundStyle:
    color: str
    text_color: str
    patch: Image.Image | None = None


def _hex(rgb):
    return '#%02x%02x%02x' % tuple(np.clip(rgb, 0, 255).astype(np.uint8))


def _blur_float(values, radius):
    radius = max(1, int(radius))
    padded = np.pad(values.astype(float), radius, mode='edge')
    summed = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    size = 2*radius+1
    return (summed[size:, size:] - summed[:-size, size:]
            - summed[size:, :-size] + summed[:-size, :-size]) / size**2



def prepare_background(frame: np.ndarray, box: tuple[int, int, int, int]) -> BackgroundStyle:
    x1, y1, x2, y2 = box
    crop = np.asarray(frame[y1:y2, x1:x2, :3], dtype=np.uint8)
    if crop.size == 0:
        return BackgroundStyle('#ffffff', '#111111')
    h, w = crop.shape[:2]
    pad = max(3, min(12, h//8))
    left, top = max(0, x1-pad), max(0, y1-pad)
    right, bottom = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
    context = np.asarray(frame[top:bottom, left:right, :3], dtype=np.uint8)
    ring = np.ones(context.shape[:2], dtype=bool)
    ring[y1-top:y2-top, x1-left:x2-left] = False
    samples = context[ring]
    if len(samples) < 8:
        samples = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    # Find a broad color cluster, without letting strokes inside the box vote
    # for the background. Use its median, not a quantization-bin center.
    bins, counts = np.unique(samples//32, axis=0, return_counts=True)
    seed = np.median(samples[np.all(samples//32 == bins[counts.argmax()], axis=1)], axis=0)
    distances = np.max(np.abs(samples.astype(float)-seed), axis=1)
    nearby = samples[distances <= 32]
    bg = np.median(nearby, axis=0)
    delta = context.astype(float)-bg
    luminance_delta = delta @ np.array([.2126, .7152, .0722])
    local_delta = luminance_delta[y1-top:y2-top, x1-left:x2-left]
    # Ink polarity comes from the stronger contrasted tail inside the row,
    # rather than whether gray paper falls above an arbitrary brightness cut.
    dark = max(0, -float(np.percentile(local_delta, 10)))
    light = max(0, float(np.percentile(local_delta, 90)))
    bg_light = dark >= light if max(dark, light) > 28 else float(np.dot(bg, [.2126, .7152, .0722])) >= 140
    ink = luminance_delta < -28 if bg_light else luminance_delta > 28
    local_ink = ink[y1-top:y2-top, x1-left:x2-left]
    foreground = crop[local_ink]
    text_color = '#111111' if bg_light else '#f5f5f5'
    if len(foreground) >= 8:
        # Estimate the core ink rather than partially transparent edge pixels.
        lum = foreground @ np.array([.2126, .7152, .0722])
        cutoff = np.percentile(lum, 30 if bg_light else 70)
        core = foreground[lum <= cutoff] if bg_light else foreground[lum >= cutoff]
        text_color = _hex(np.median(core, axis=0))
    color = _hex(bg)
    spread = np.percentile(np.max(np.abs(samples.astype(float)-bg), axis=1), 80)
    if spread <= 4:
        return BackgroundStyle(color, text_color)
    mask = Image.fromarray((ink*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    valid = 1-np.asarray(mask, dtype=float)/255
    # Normalized convolution excludes ink from every color average. This
    # prevents the dark panels produced by blurring bold source letters.
    radius = max(3, h*.35)
    weights = _blur_float(valid, radius)
    channels = []
    for channel in range(3):
        total = _blur_float(context[..., channel]*valid, radius)
        channels.append(np.divide(total, weights, out=np.full_like(total, bg[channel]), where=weights > .01))
    clean = np.stack(channels, axis=2)
    alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(.7)), dtype=float)/255
    alpha = np.maximum(alpha, np.asarray(mask, dtype=float)/255)
    # Preserve the untouched texture; soften only the erased glyph area.
    repaired = context*(1-alpha[..., None]) + clean*alpha[..., None]
    patch = repaired[y1-top:y2-top, x1-left:x2-left]
    return BackgroundStyle(color, text_color, Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8)))
