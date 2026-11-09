"""Conservative corrections before filtering, preserving decorative markers."""
from dataclasses import replace
import re
import numpy as np
from modules.text_processing.filters import is_contact_address, is_ui_label


def _components(mask):
    """Connected run components; no optional imaging dependencies required."""
    parents, bounds, previous = [], [], []
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        current = []
        for start, end in zip(edges[::2], edges[1::2]):
            hits = {root(i) for a, b, i in previous if a <= end and b >= start}
            if hits:
                i = min(hits)
                for j in hits - {i}:
                    parents[j] = i
                    a = bounds[j]
                    b = bounds[i]
                    bounds[i] = [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]), a[4]+b[4]]
                b = bounds[i]
                bounds[i] = [min(b[0],start), b[1], max(b[2],end), y+1, b[4]+end-start]
            else:
                i = len(parents)
                parents.append(i)
                bounds.append([start,y,end,y+1,end-start])
            current.append((start,end,i))
        previous = current
    return [bounds[i] for i in range(len(parents)) if root(i) == i]


def refine_box(image, line, neighbors=()):
    """Recover glyph extents, excluding broad picture strips and viewer edges."""
    height, width = image.shape[:2]
    x1, x2 = max(0,line.left-3), min(width,line.right+3)
    y1, y2 = max(0,line.top-4), min(height,line.bottom+4)
    center = (line.top+line.bottom)/2
    for other in neighbors:
        if other is line or min(line.right, other.right) <= max(line.left, other.left):
            continue
        other_center = (other.top+other.bottom)/2
        if other_center > center:
            y2 = min(y2, max(line.bottom, other.top))
        elif other_center < center:
            y1 = max(y1, min(line.top, other.bottom))
    crop = image[y1:y2,x1:x2,:3].astype(float)
    if not crop.size:
        return line
    gray = crop.mean(axis=2)
    # Refine only light/medium paper. Dark UI text needs a separate polarity.
    # Estimate the paper from the border, not from bright glyphs filling a
    # tight OCR box. White text on a photograph must not reverse polarity.
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    bg = float(np.median(border))
    if bg < 90 or np.percentile(gray, 1) > bg-15:
        return line
    mask = gray < bg-max(15, (bg-float(np.percentile(gray, 10)))*.35)
    h, w = mask.shape
    parts = [b for b in _components(mask)
             if b[4] >= 3 and b[2]-b[0] < w*.75
             and b[3]-b[1] < h*.95
             and b[3] > h*.18 and b[1] < h*.82]
    if len(parts) < 2:
        return line
    # Only extend by the search margin; never swallow a neighbouring row.
    left = max(x1, x1+min(b[0] for b in parts)-1)
    right = min(x2, x1+max(b[2] for b in parts)+1)
    top = max(y1, y1+min(b[1] for b in parts)-1)
    bottom = min(y2, y1+max(b[3] for b in parts)+1)
    if right-left < (line.right-line.left)*.5:
        return line
    # OCR owns the source extent. Pixel refinement may recover clipped edges,
    # but must never discard the beginning or end of a recognized word.
    if line.polygon is None:
        left, right = min(left, line.left), max(right, line.right)
        top, bottom = min(top, line.top), max(bottom, line.bottom)
    return replace(line, left=int(left), top=int(top), right=int(right), bottom=int(bottom))


def prepare_lines(image, lines):
    result = []
    for line in sorted(lines, key=lambda item: (item.top, item.left)):
        # Known UI tokens must not be merged with their neighbouring labels.
        if is_ui_label(line.text):
            continue
        line = refine_box(image, line, lines)
        text = re.sub(r'^[»›》>]+\s*', '', line.text)
        if line.right > line.left:
            crop = image[line.top:line.bottom, line.left:line.right, :3].astype(float)
            limit = min(crop.shape[1], max(8, (line.bottom-line.top)*2))
            part = crop[:, :limit]
            warm = ((part[..., 0] > part[..., 1]*1.2) & (part[..., 0] > part[..., 2]*1.2)
                    & (part[..., 0]-part[..., 1] > 40) & (part[..., 0]-part[..., 2] > 40))
            cols = np.flatnonzero(warm.sum(axis=0) >= 2)
            # Detect a detached warm marker even if OCR omitted its symbol.
            if (len(cols) and cols[-1] < limit-3
                    and (text != line.text or
                         (np.percentile(part.mean(axis=2), 25) > 140
                          and cols[-1]-cols[0]+1 <= (line.bottom-line.top)*.65))):
                start = int(cols[-1])+3
                ink = np.max(np.abs(crop-np.percentile(crop, 85, axis=(0,1))), axis=2) > 30
                following = np.flatnonzero(ink[:, start:].sum(axis=0) >= 2)
                if len(following):
                    start += max(0, int(following[0])-1)
                line = replace(line, left=line.left+start, text=text)
        result.append(line)
    # Windows sometimes returns an isolated drop-cap/article beside its line.
    for i in range(len(result)):
        first = result[i]
        if not first.text or not first.text.strip().isalpha() or len(first.text.strip()) > 3:
            continue
        h = first.bottom-first.top
        for j in range(len(result)):
            if j == i:
                continue
            other = result[j]
            if not other.text:
                continue
            overlap = min(first.bottom, other.bottom)-max(first.top, other.top)
            if overlap >= .7*min(h, other.bottom-other.top) and 0 <= other.left-first.right <= h:
                result[j] = replace(other, text=first.text+' '+other.text,
                                    left=first.left, top=min(first.top,other.top),
                                    bottom=max(first.bottom,other.bottom))
                result[i] = replace(first, text='')
                break
    # Contact captions may share weight, but never borrow address geometry.
    captions = []
    contacts = [line for line in result if is_contact_address(line.text)]
    for i, line in enumerate(result):
        if not line.text or is_contact_address(line.text):
            continue
        column = [other for other in contacts if abs(other.left-line.left) <= 6]
        below = [other for other in column
                 if 0 <= other.top-line.bottom <= (line.bottom-line.top)*1.5]
        if not below:
            continue
        captions.append(i)
    if len(captions) >= 3:
        from modules.overlay.styling import estimate_bold
        weight = bool(sum(estimate_bold(image,result[i]) for i in captions) >= len(captions)/2)
        for i in captions:
            result[i] = replace(result[i], bold=weight)
    return [line for line in result if line.text]
