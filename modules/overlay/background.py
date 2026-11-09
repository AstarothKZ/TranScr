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
    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
        raise ValueError(f"Background box outside frame: {box}")
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
    # The outside ring may belong to a viewer's black surround or the next
    # row. Give the margin inside this OCR box the deciding vote.
    rim = np.ones((h, w), dtype=bool)
    inset = max(1, min(3, h // 8, w // 8))
    rim[inset:-inset, inset:-inset] = False
    inner_samples = crop[rim]
    outside = context[ring]
    step = max(1, len(outside) // max(1, len(inner_samples) // 4))
    samples = np.concatenate((inner_samples, outside[::step]))
    if len(samples) < 8:
        samples = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    # Find a broad color cluster, without letting strokes inside the box vote
    # for the background. Use its median, not a quantization-bin center.
    bins, counts = np.unique(samples//32, axis=0, return_counts=True)
    seed = np.median(samples[np.all(samples//32 == bins[counts.argmax()], axis=1)], axis=0)
    distances = np.max(np.abs(samples.astype(float)-seed), axis=1)
    nearby = samples[distances <= 32]
    bg = np.median(nearby, axis=0)
    # Tight light labels can have more ink than background in their inner rim.
    # Trust a uniform exterior when the box corners agree with it. This avoids
    # white panels behind white footer links without mistaking paper surrounded
    # by a dark image viewer for a dark text background.
    flat_exterior = False
    if len(outside) >= 8:
        exterior = np.median(outside, axis=0)
        corners = crop[[0, 0, -1, -1], [0, -1, 0, -1]].astype(float)
        flat_exterior = bool(
            np.percentile(np.max(np.abs(outside.astype(float)-exterior), axis=1), 90) <= 4
            and np.sum(np.max(np.abs(corners-exterior), axis=1) <= 12) >= 3
        )
        if flat_exterior:
            bg = exterior
    # A large detection can include the viewer surround in its own margins.
    # Require substantial interior evidence before overturning a dark border.
    interior = crop.reshape(-1, 3).astype(float)
    brighter = np.min(interior-bg, axis=1) > 40
    if not flat_exterior and np.mean(bg) < 70 and brighter.mean() > .35:
        from modules.text_processing.geometry import _components
        # Paper is one broad connected surface; white letters are separate
        # components. Density alone cannot distinguish the two.
        surfaces = _components(brighter.reshape(h, w))
        if any(b[2]-b[0] >= w*.8 and b[4] >= h*w*.25 for b in surfaces):
            bg = np.median(interior[brighter], axis=0)
    delta = context.astype(float)-bg
    luminance_delta = delta @ np.array([.2126, .7152, .0722])
    local_delta = luminance_delta[y1-top:y2-top, x1-left:x2-left]
    # Ink polarity comes from the stronger contrasted tail inside the row,
    # rather than whether gray paper falls above an arbitrary brightness cut.
    dark = max(0, -float(np.percentile(local_delta, 10)))
    light = max(0, float(np.percentile(local_delta, 90)))
    contrast = max(dark, light)
    bg_light = dark >= light if contrast > 8 else float(np.dot(bg, [.2126, .7152, .0722])) >= 140
    threshold = max(6, min(28, contrast * .35))
    ink = luminance_delta < -threshold if bg_light else luminance_delta > threshold
    local_ink = ink[y1-top:y2-top, x1-left:x2-left]
    foreground = crop[local_ink]
    text_color = '#111111' if bg_light else '#f5f5f5'
    if len(foreground) >= 8:
        # Estimate the core ink rather than partially transparent edge pixels.
        lum = foreground @ np.array([.2126, .7152, .0722])
        cutoff = np.percentile(lum, 30 if bg_light else 70)
        core = foreground[lum <= cutoff] if bg_light else foreground[lum >= cutoff]
        text_color = _hex(np.median(core, axis=0))
        ink_rgb = np.median(core, axis=0)
        if np.ptp(ink_rgb) > 35:
            # Colored lettering over a pale watermark: erase matching pigment,
            # not every darker pixel of the underlying illustration.
            direction = bg-ink_rgb
            length = float(direction @ direction)
            amount = np.clip((-delta @ direction) / max(1, length), 0, 1)
            predicted = bg-amount[..., None]*direction
            pigment = (np.max(np.abs(context-predicted), axis=2) < 25) & (amount > .12)
            # A line may mix colored words with black words (e.g. GAME BLOG).
            dark_core = luminance_delta < -max(50, contrast*.65)
            ink &= pigment | (dark_core if bg_light else False)
            local_ink = ink[y1-top:y2-top, x1-left:x2-left]
    color = _hex(bg)
    spread = np.percentile(np.max(np.abs(samples.astype(float)-bg), axis=1), 80)
    remaining = crop[~local_ink]
    interior_flat = not len(remaining) or np.percentile(np.max(np.abs(remaining.astype(float)-bg), axis=1), 95) <= 4
    if spread <= 4 and interior_flat:
        return BackgroundStyle(color, text_color)
    mask = Image.fromarray((ink*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    valid = 1-np.asarray(mask, dtype=float)/255
    # Normalized convolution excludes ink from every color average. This
    # prevents the dark panels produced by blurring bold source letters.
    radius = max(3, min(64, h*.35))
    weights = _blur_float(valid, radius)
    # Where a thick glyph leaves no nearby samples, interpolate a smooth
    # background from the unmasked context instead of a flat median rectangle.
    yy, xx = np.indices(valid.shape, dtype=float)
    xx /= max(1, valid.shape[1] - 1)
    yy /= max(1, valid.shape[0] - 1)
    usable = (valid > .99) & (np.max(np.abs(delta), axis=2) <= 32)
    positions = np.flatnonzero(usable)[::max(1, int(usable.sum()) // 2048)]
    plane = None
    if len(positions) >= 8:
        design = np.column_stack((np.ones(len(positions)), xx.flat[positions], yy.flat[positions]))
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, context.reshape(-1, 3)[positions], rcond=None
        )
        if rank == 3:
            plane = coefficients[0] + xx[..., None]*coefficients[1] + yy[..., None]*coefficients[2]
            plane = np.clip(plane, np.maximum(0, bg-32), np.minimum(255, bg+32))
    channels = []
    for channel in range(3):
        total = _blur_float(context[..., channel]*valid, radius)
        fallback = plane[..., channel].copy() if plane is not None else np.full_like(total, bg[channel])
        channels.append(np.divide(total, weights, out=fallback, where=weights > .01))
    clean = np.stack(channels, axis=2)
    alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(.7)), dtype=float)/255
    alpha = np.maximum(alpha, np.asarray(mask, dtype=float)/255)
    # Preserve the untouched texture; soften only the erased glyph area.
    repaired = context*(1-alpha[..., None]) + clean*alpha[..., None]
    if spread <= 18 and contrast > 30 and float(local_ink.mean()) > .20 and interior_flat:
        # Dense display headings on paper leave letter-shaped smooth islands
        # if only the ink is repaired. Rebuild the whole smooth surface using
        # its fitted gradient and measured paper grain, without glyph outlines.
        smoothed = np.asarray(Image.fromarray(context).filter(ImageFilter.GaussianBlur(1)), dtype=float)
        safe = np.asarray(Image.fromarray((usable * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))) > 0
        residual = (context.astype(float) - smoothed)[safe]
        residual = residual[np.max(np.abs(residual), axis=1) <= 12]
        if len(residual) >= 8:
            residual = residual - np.mean(residual, axis=0)
            rng = np.random.default_rng(0)
            grain = residual[rng.integers(0, len(residual), size=context.shape[:2])]
            repaired = clean + grain
            # Join the patch to the source at clean border pixels. Ink remains
            # fully erased even when an OCR rectangle touches a letter edge.
            px = np.arange(context.shape[1])[None, :]
            py = np.arange(context.shape[0])[:, None]
            edge = np.minimum(
                np.minimum(px - (x1-left), (x2-left-1) - px),
                np.minimum(py - (y1-top), (y2-top-1) - py),
            )
            blend = np.maximum(np.clip(edge / 6, 0, 1), np.asarray(mask, dtype=float) / 255)
            repaired = context*(1-blend[..., None]) + repaired*blend[..., None]
    patch = repaired[y1-top:y2-top, x1-left:x2-left]
    return BackgroundStyle(color, text_color, Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8)))
