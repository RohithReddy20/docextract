from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class Affine:
    """x' = a*x + c*y + e; y' = b*x + d*y + f."""

    a: float = 1
    b: float = 0
    c: float = 0
    d: float = 1
    e: float = 0
    f: float = 0

    def point(self, point: Point) -> Point:
        x, y = point
        return self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f

    def polygon(self, polygon: Polygon) -> Polygon:
        return tuple(self.point(point) for point in polygon)

    def then(self, other: Affine) -> Affine:
        return Affine(
            other.a * self.a + other.c * self.b,
            other.b * self.a + other.d * self.b,
            other.a * self.c + other.c * self.d,
            other.b * self.c + other.d * self.d,
            other.a * self.e + other.c * self.f + other.e,
            other.b * self.e + other.d * self.f + other.f,
        )

    def inverse(self) -> Affine:
        determinant = self.a * self.d - self.b * self.c
        if determinant == 0:
            raise ValueError("Singular coordinate transform")
        return Affine(
            self.d / determinant,
            -self.b / determinant,
            -self.c / determinant,
            self.a / determinant,
            (self.c * self.f - self.d * self.e) / determinant,
            (self.b * self.e - self.a * self.f) / determinant,
        )

    def values(self) -> tuple[float, float, float, float, float, float]:
        return self.a, self.b, self.c, self.d, self.e, self.f


def page_transform(crop: tuple[float, float, float, float], rotation: int) -> tuple[Affine, float, float]:
    left, bottom, right, top = crop
    width, height = right - left, top - bottom
    if width <= 0 or height <= 0:
        raise ValueError("Invalid crop box")
    transforms = {
        0: Affine(1, 0, 0, -1, -left, top),
        90: Affine(0, 1, 1, 0, -bottom, -left),
        180: Affine(-1, 0, 0, 1, right, -bottom),
        270: Affine(0, -1, -1, 0, top, right),
    }
    if rotation not in transforms:
        raise ValueError("Page rotation must be 0, 90, 180 or 270")
    return transforms[rotation], *((height, width) if rotation % 180 else (width, height))


def crop_resize(left: float, top: float, scale_x: float, scale_y: float) -> Affine:
    if min(scale_x, scale_y) <= 0:
        raise ValueError("Crop scales must be positive")
    return Affine(scale_x, 0, 0, scale_y, -left * scale_x, -top * scale_y)


def rectangle(box: tuple[float, float, float, float]) -> Polygon:
    left, bottom, right, top = box
    return ((left, bottom), (right, bottom), (right, top), (left, top))


def bounds(polygon: Polygon) -> tuple[float, float, float, float]:
    xs, ys = zip(*polygon, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def envelope(polygons: list[Polygon]) -> Polygon:
    return rectangle(bounds(tuple(point for polygon in polygons for point in polygon)))


def normalized_coco(polygon: Polygon, width: float, height: float) -> list[float]:
    x0, y0, x1, y1 = bounds(polygon)
    x0, x1 = max(0.0, x0), min(width, x1)
    y0, y1 = max(0.0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Evidence lies outside the visible page")
    return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]
