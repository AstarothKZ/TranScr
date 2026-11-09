"""Recover independent small light navigation links inside a merged OCR row."""
from dataclasses import replace
import numpy as np
from PIL import Image, ImageOps


def recover_labels(image, lines, recognize):
    output = []
    replaced = []
    for line in lines:
        if any(a <= (line.left+line.right)/2 <= c
               and b <= (line.top+line.bottom)/2 <= d for a,b,c,d in replaced):
            continue
        h = line.bottom-line.top
        if h < 6 or line.right-line.left < h*18:
            continue
        x1, y1 = max(0,line.left-3), max(0,line.top-3)
        x2, y2 = min(image.shape[1],line.right+5*h), min(image.shape[0],line.bottom+3)
        crop = image[y1:y2, x1:x2, :3]
        gray = crop.astype(float).mean(axis=2)
        if np.median(gray) > 80:
            continue
        columns = (gray > max(110, float(np.median(gray))+65)).sum(axis=0) >= 2
        edges = np.flatnonzero(np.diff(np.r_[False,columns,False]))
        groups = []
        for a,b in zip(edges[::2], edges[1::2]):
            if groups and a-groups[-1][1] < h*.55:
                groups[-1][1] = b
            else:
                groups.append([int(a),int(b)])
        groups = [(a,b) for a,b in groups if b-a >= h*.65]
        if not 3 <= len(groups) <= 16:
            continue
        recovered = []
        for a, b in groups:
            # Keep the detected ink extent; whitespace is not layout capacity.
            left, right = max(0, a-1), min(crop.shape[1], b+1)
            # OCR needs breathing room around short condensed white labels.
            recognition_left, recognition_right = max(0,a-2), min(crop.shape[1],b+2)
            padded = ImageOps.expand(Image.fromarray(crop[:,recognition_left:recognition_right]),
                border=12, fill=tuple(np.median(crop.reshape(-1,3),axis=0).astype(int)))
            enlarged = np.array(padded.resize(
                (padded.width*3,padded.height*3), Image.Resampling.LANCZOS))
            items = recognize(enlarged)
            if not items:
                recovered = []
                break
            # A recovery must account for every detected label. Partial OCR
            # must never discard the original row or unrelated source words.
            recovered.append(replace(items[0], text=" ".join(i.text for i in items),
                left=int(x1+left), right=int(x1+right), top=line.top,
                bottom=line.bottom, polygon=None, label_size=None))
        if recovered:
            replaced.append((x1,y1,x2,y2))
            output.extend(recovered)
    # Drop the original merged row and any duplicates covered by its recovery.
    return output + [l for l in lines if not any(
        a <= (l.left+l.right)/2 <= c and b <= (l.top+l.bottom)/2 <= d
        for a,b,c,d in replaced)]
