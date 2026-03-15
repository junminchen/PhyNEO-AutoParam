#!/usr/bin/env python3
import argparse
import csv
import struct
import zlib
from pathlib import Path


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "batch": r["batch"],
                    "distance": float(r["distance"]),
                    "calc_lr_es": float(r["calc_lr_es"]),
                    "calc_lr_pol": float(r["calc_lr_pol"]),
                    "calc_lr_disp": float(r["calc_lr_disp"]),
                    "calc_lr_tot": float(r["calc_lr_tot"]),
                    "ref_lr_es": float(r["ref_lr_es"]),
                    "ref_lr_pol": float(r["ref_lr_pol"]),
                    "ref_lr_disp": float(r["ref_lr_disp"]),
                    "ref_lr_tot": float(r["ref_lr_tot"]),
                }
            )
    return rows


class Canvas:
    def __init__(self, w: int, h: int, bg=(248, 248, 248)):
        self.w = w
        self.h = h
        self.buf = bytearray([0] * (w * h * 3))
        self.fill(bg)

    def fill(self, c):
        r, g, b = c
        for y in range(self.h):
            row = y * self.w * 3
            for x in range(self.w):
                i = row + x * 3
                self.buf[i] = r
                self.buf[i + 1] = g
                self.buf[i + 2] = b

    def set_px(self, x, y, c):
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return
        i = (y * self.w + x) * 3
        self.buf[i] = c[0]
        self.buf[i + 1] = c[1]
        self.buf[i + 2] = c[2]

    def line(self, x0, y0, x1, y1, c):
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_px(x0, y0, c)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(self, x0, y0, x1, y1, c):
        self.line(x0, y0, x1, y0, c)
        self.line(x1, y0, x1, y1, c)
        self.line(x1, y1, x0, y1, c)
        self.line(x0, y1, x0, y0, c)

    def point(self, x, y, c, r=1):
        for yy in range(y - r, y + r + 1):
            for xx in range(x - r, x + r + 1):
                self.set_px(xx, yy, c)

    def save_png(self, path: Path):
        raw = bytearray()
        row_bytes = self.w * 3
        for y in range(self.h):
            raw.append(0)  # filter type 0
            start = y * row_bytes
            raw.extend(self.buf[start : start + row_bytes])
        comp = zlib.compress(bytes(raw), level=6)

        def chunk(tag: bytes, data: bytes):
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = bytearray()
        png.extend(b"\x89PNG\r\n\x1a\n")
        png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0)))
        png.extend(chunk(b"IDAT", comp))
        png.extend(chunk(b"IEND", b""))
        with open(path, "wb") as f:
            f.write(png)


def draw_panel(cv: Canvas, box, rows, calc_key, ref_key):
    x0, y0, x1, y1 = box
    pad = 28
    px0, py0 = x0 + pad, y0 + pad
    px1, py1 = x1 - pad, y1 - pad
    cv.rect(x0, y0, x1, y1, (140, 140, 140))
    cv.line(px0, py0, px0, py1, (100, 100, 100))
    cv.line(px0, py1, px1, py1, (100, 100, 100))

    xs = [r["distance"] for r in rows]
    ys = [r[calc_key] for r in rows] + [r[ref_key] for r in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0

    def tx(x):
        return int(px0 + (x - x_min) / (x_max - x_min) * (px1 - px0))

    def ty(y):
        return int(py1 - (y - y_min) / (y_max - y_min) * (py1 - py0))

    ordered = sorted(rows, key=lambda r: r["distance"])

    # calc: blue line
    prev = None
    for r in ordered:
        p = (tx(r["distance"]), ty(r[calc_key]))
        if prev is not None:
            cv.line(prev[0], prev[1], p[0], p[1], (36, 92, 172))
        prev = p

    # ref: orange line
    prev = None
    for r in ordered:
        p = (tx(r["distance"]), ty(r[ref_key]))
        if prev is not None:
            cv.line(prev[0], prev[1], p[0], p[1], (224, 126, 44))
        prev = p


def main():
    parser = argparse.ArgumentParser(description="Plot dimer LR comparison CSV to PNG (stdlib only).")
    parser.add_argument("--csv", default="ec_ec_lr_compare.csv")
    parser.add_argument("--out", default="ec_ec_lr_compare.png")
    parser.add_argument("--batch", default="000", help="Only plot one batch id")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv).resolve())
    rows = [r for r in rows if r["batch"] == args.batch]
    if not rows:
        raise ValueError(f"CSV has no rows for batch={args.batch}.")

    w, h = 1600, 1000
    cv = Canvas(w, h, bg=(248, 248, 248))

    margin = 30
    top = 30
    gap_x = 20
    gap_y = 20
    panel_w = (w - 2 * margin - gap_x) // 2
    panel_h = (h - top - margin - gap_y) // 2
    boxes = [
        (margin, top, margin + panel_w, top + panel_h),
        (margin + panel_w + gap_x, top, margin + 2 * panel_w + gap_x, top + panel_h),
        (margin, top + panel_h + gap_y, margin + panel_w, top + 2 * panel_h + gap_y),
        (
            margin + panel_w + gap_x,
            top + panel_h + gap_y,
            margin + 2 * panel_w + gap_x,
            top + 2 * panel_h + gap_y,
        ),
    ]

    draw_panel(cv, boxes[0], rows, "calc_lr_tot", "ref_lr_tot")
    draw_panel(cv, boxes[1], rows, "calc_lr_es", "ref_lr_es")
    draw_panel(cv, boxes[2], rows, "calc_lr_pol", "ref_lr_pol")
    draw_panel(cv, boxes[3], rows, "calc_lr_disp", "ref_lr_disp")

    out = Path(args.out).resolve()
    cv.save_png(out)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
