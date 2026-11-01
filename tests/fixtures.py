"""Synthetic test assets.

These exist so the pipeline can be verified end to end without a physical chart,
and so correctness is checkable by construction: the reference values and the
rendered image are generated from the same numbers, so a clean round trip must
return Delta E at the quantisation floor. Any error anywhere in ICC parsing,
matrix maths, Lab conversion, homography or sampling shows up immediately.

Nothing here is presented as real measurement data. Files written by
``write_reference`` carry an explicit synthetic marker in their DESCRIPTOR.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deltachrome.imaging.color import lab_to_xyz
from deltachrome.imaging.icc import CIE_D50

# -- a real, minimal, matrix/TRC ICC profile --------------------------------

ADOBE_RGB_PRIMARIES = np.array([[0.6400, 0.3300], [0.2100, 0.7100], [0.1500, 0.0600]])
D65_XY = (0.3127, 0.3290)
ADOBE_RGB_GAMMA = 563.0 / 256.0
#: The gamma of the monochrome working space these fixtures use. Gray Gamma 2.2 is
#: on FADGI's permitted list for microfilm, which is where grayscale masters come
#: from in this domain.
GRAY_GAMMA = 2.2

_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def rgb_to_xyz_matrix(primaries: np.ndarray, white_xy: tuple[float, float]) -> np.ndarray:
    """The RGB -> XYZ matrix for a set of primaries and a white point."""
    p = np.column_stack([_xy_to_xyz(x, y) for x, y in primaries])
    white = _xy_to_xyz(*white_xy)
    scale = np.linalg.solve(p, white)
    return p * scale


def bradford_adapt_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Bradford chromatic adaptation matrix between two white points (XYZ)."""
    s = _BRADFORD @ source
    t = _BRADFORD @ target
    return np.linalg.inv(_BRADFORD) @ np.diag(t / s) @ _BRADFORD


def adobe_rgb_matrix() -> np.ndarray:
    """Adobe RGB (1998) RGB -> XYZ, Bradford-adapted to the D50 PCS."""
    native = rgb_to_xyz_matrix(ADOBE_RGB_PRIMARIES, D65_XY)
    adapt = bradford_adapt_matrix(_xy_to_xyz(*D65_XY), np.array(CIE_D50))
    return adapt @ native


def _s15f16(value: float) -> bytes:
    return struct.pack(">i", int(round(value * 65536.0)))


def _tag_xyz(xyz: np.ndarray) -> bytes:
    return b"XYZ " + b"\x00" * 4 + b"".join(_s15f16(float(v)) for v in xyz)


def _tag_curve_gamma(gamma: float) -> bytes:
    return b"curv" + b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(gamma * 256)))


def _tag_desc(text: str) -> bytes:
    raw = text.encode("ascii") + b"\x00"
    return (
        b"desc"
        + b"\x00" * 4
        + struct.pack(">I", len(raw))
        + raw
        + b"\x00" * 12  # unicode language code + count + scriptcode header
        + b"\x00" * 67  # macintosh scriptcode buffer
    )


def build_matrix_icc(
    matrix: np.ndarray,
    gamma: float,
    description: str,
    white_point: tuple[float, float, float] = CIE_D50,
) -> bytes:
    """Assemble a minimal but structurally valid ICC v2 matrix/TRC RGB profile."""
    tags: list[tuple[bytes, bytes]] = [
        (b"desc", _tag_desc(description)),
        (b"wtpt", _tag_xyz(np.array(white_point))),
        (b"rXYZ", _tag_xyz(matrix[:, 0])),
        (b"gXYZ", _tag_xyz(matrix[:, 1])),
        (b"bXYZ", _tag_xyz(matrix[:, 2])),
        (b"rTRC", _tag_curve_gamma(gamma)),
        (b"gTRC", _tag_curve_gamma(gamma)),
        (b"bTRC", _tag_curve_gamma(gamma)),
    ]

    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    entries: list[bytes] = []
    body = bytearray()
    for sig, payload in tags:
        pad = (-len(payload)) % 4
        entries.append(struct.pack(">4sII", sig, offset, len(payload)))
        body += payload + b"\x00" * pad
        offset += len(payload) + pad

    total = 128 + table_size + len(body)
    header = bytearray(128)
    struct.pack_into(">I", header, 0, total)
    header[4:8] = b"none"          # preferred CMM
    struct.pack_into(">I", header, 8, 0x02400000)  # version 2.4
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "
    header[36:40] = b"acsp"
    struct.pack_into(">I", header, 64, 1)  # rendering intent: relative colorimetric
    header[68:80] = _tag_xyz(np.array(CIE_D50))[8:20]

    return bytes(header) + struct.pack(">I", len(tags)) + b"".join(entries) + bytes(body)


def adobe_rgb_icc() -> bytes:
    return build_matrix_icc(
        adobe_rgb_matrix(), ADOBE_RGB_GAMMA, "Synthetic Adobe RGB (1998) [Deltachrome test]"
    )


def build_gray_icc(gamma: float = 2.2, description: str = "Gray Gamma 2.2") -> bytes:
    """A minimal monochrome ICC profile: one tone curve, no colorants.

    Structurally different from an RGB profile rather than a special case of one --
    ``GRAY`` in the colour space field, and ``kTRC`` where an RGB profile carries six
    tags. That is what makes it worth building here rather than faking: the reader
    has to pick the branch from the file, not from how many channels the pixels
    happen to have.
    """
    tags: list[tuple[bytes, bytes]] = [
        (b"desc", _tag_desc(description)),
        (b"wtpt", _tag_xyz(np.array(CIE_D50))),
        (b"kTRC", _tag_curve_gamma(gamma)),
    ]
    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    entries: list[bytes] = []
    body = bytearray()
    for sig, payload in tags:
        pad = (-len(payload)) % 4
        entries.append(struct.pack(">4sII", sig, offset, len(payload)))
        body += payload + b"\x00" * pad
        offset += len(payload) + pad

    total = 128 + table_size + len(body)
    header = bytearray(128)
    struct.pack_into(">I", header, 0, total)
    header[4:8] = b"none"
    struct.pack_into(">I", header, 8, 0x02400000)
    header[12:16] = b"mntr"
    header[16:20] = b"GRAY"
    header[20:24] = b"XYZ "
    header[36:40] = b"acsp"
    struct.pack_into(">I", header, 64, 1)
    header[68:80] = _tag_xyz(np.array(CIE_D50))[8:20]
    return bytes(header) + struct.pack(">I", len(tags)) + b"".join(entries) + bytes(body)


# -- synthetic ColorChecker SG ---------------------------------------------

COLS = list("ABCDEFGHIJKLMN")
ROWS = list(range(1, 11))
RAMP_IDS = [f"{c}5" for c in "EFGHIJ"] + [f"{c}6" for c in "EFGHIJ"]
#: The repeated near-white perimeter neutrals, matching the primer's description
#: of A1 (~L96, a-0.9, b3.2) recurring at multiple border positions.
UNIFORMITY_IDS = [
    "A1", "D1", "H1", "L1", "N1",
    "A4", "N4",
    "A7", "N7",
    "A10", "E10", "I10", "N10",
]


@dataclass
class SyntheticChart:
    """Deterministic reference values for a fake 140-patch SG."""

    seed: int = 20260818

    def reference_lab(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        values: dict[str, np.ndarray] = {}

        for i, patch_id in enumerate(RAMP_IDS):
            L = 96.0 - i * (96.0 - 10.0) / (len(RAMP_IDS) - 1)
            values[patch_id] = np.array([L, -0.4, 0.6])

        for patch_id in UNIFORMITY_IDS:
            # Near-identical, with the small real spread a physical chart has.
            values[patch_id] = np.array([96.0, -0.9, 3.2]) + rng.normal(0, 0.06, 3)

        # Chromatic patches are generated in *device* space and converted to Lab,
        # rather than the other way round. That guarantees every reference value is
        # inside the working space's gamut, so a clean render round-trips exactly.
        # Generating Lab directly would put some patches outside Adobe RGB, and the
        # resulting clipping would be a property of the fixture rather than of
        # anything under test.
        from deltachrome.imaging.color import xyz_to_lab

        matrix = adobe_rgb_matrix()
        for row in ROWS:
            for col in COLS:
                patch_id = f"{col}{row}"
                if patch_id in values:
                    continue
                device = rng.uniform(0.06, 0.94, 3)
                linear = device**ADOBE_RGB_GAMMA
                values[patch_id] = xyz_to_lab(matrix @ linear, white_point=CIE_D50)
        return values

    def write_reference(self, path: Path, *, duplicate: str | None = None) -> Path:
        """Write a CGATS file. ``duplicate`` injects a conflicting repeated sample."""
        labs = self.reference_lab()
        lines = [
            "CGATS.17",
            'ORIGINATOR "Deltachrome synthetic fixture"',
            'DESCRIPTOR "SYNTHETIC TEST DATA -- NOT A REAL MEASUREMENT"',
            'CREATED "2026-08-18"',
            'INSTRUMENTATION "synthetic"',
            'MEASUREMENT_SOURCE "Illumination=M1 ObserverAngle=2"',
            'ILLUMINATION_NAME "D50"',
            'OBSERVER_ANGLE "2"',
            'DEVCALSTD "XRDI"',
            "",
            "NUMBER_OF_FIELDS 4",
            "BEGIN_DATA_FORMAT",
            "SAMPLE_NAME LAB_L LAB_A LAB_B",
            "END_DATA_FORMAT",
            "",
        ]
        rows = [
            f"{pid} {lab[0]:.4f} {lab[1]:.4f} {lab[2]:.4f}"
            for pid, lab in sorted(labs.items(), key=_patch_sort_key)
        ]
        if duplicate:
            lab = labs[duplicate]
            rows.append(f"{duplicate} {lab[0] + 7.5:.4f} {lab[1] - 3.0:.4f} {lab[2] + 2.0:.4f}")
        lines.append(f"NUMBER_OF_SETS {len(rows)}")
        lines.append("BEGIN_DATA")
        lines.extend(rows)
        lines.append("END_DATA")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def render_capture(
        self,
        path: Path,
        *,
        # Large enough to be a plausible capture rather than merely a picture of
        # a chart. A ColorChecker SG is 292 mm across, so 1400 px is 128 ppi --
        # under half FADGI's 3-star requirement, and the round-trip tests were
        # asserting a conformance PASS on it only because the profile they used
        # did not yet declare a sampling-frequency metric. At this size it is
        # 314 ppi, which passes on its merits, and it costs about 0.14 s a test.
        size: tuple[int, int] = (3200, 2300),
        margin: int = 60,
        gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
        gamma_error: float = 1.0,
        skew: int = 0,
        icc: bytes | None = None,
        grayscale: bool = False,
        ruler: bool = True,
    ) -> tuple[Path, list[tuple[float, float]]]:
        r"""Render the chart to a 16-bit TIFF with an Adobe RGB profile embedded.

        ``gain`` and ``gamma_error`` inject a known capture error; the defaults
        render an exact chart, which must come back at the quantisation floor.
        ``skew`` shifts the top edge to exercise the perspective transform.

        ``ruler`` draws the millimetre scale the SG prints below columns K-N, in the
        band the target definition declares, on rows added BELOW ``size`` so the
        fiducial quad and every existing sampling rate stay exactly as they were. It
        is on by default because the chart being modelled prints one, and a fixture
        that omits a declared feature makes the metrics reading that feature look
        broken. Pass ``ruler=False`` for the case of a capture that crops it away.

        ``grayscale`` renders the same chart as a **single colour record**, tagged
        with a matching monochrome profile: each patch becomes its own relative
        luminance through a gamma 2.2 curve. That is a real grayscale capture of a
        colour chart, which is the awkward case worth having -- every neutral still
        round-trips to its reference L\*, and every chromatic patch has lost the
        information a colour metric would need, without anything in the file saying
        so.
        """
        import tifffile

        from deltachrome.targets.registry import load_target

        target = load_target("colorchecker_sg_v1")
        grid = target.patch_grid
        assert grid is not None

        labs = self.reference_lab()
        matrix = adobe_rgb_matrix()
        inverse = np.linalg.inv(matrix)

        width, height = size
        # Extra rows below the quad, so the printed scale at v 1.02-1.05 has
        # somewhere to be drawn. The quad itself is unchanged.
        foot = int(round(0.10 * (height - 2 * margin))) if ruler else 0
        canvas = np.zeros((height + foot, width, 3), dtype=np.float64)

        corners = [
            (margin + skew, margin),
            (width - margin, margin),
            (width - margin - skew, height - margin),
            (margin, height - margin),
        ]
        src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        import cv2

        homography = cv2.getPerspectiveTransform(src, np.array(corners, dtype=np.float32))

        if grayscale:
            # A fresh array, not ``canvas[..., :1]``: that view is not contiguous
            # and cv2.fillPoly refuses it.
            canvas = np.zeros((height + foot, width, 1), dtype=np.float64)

        for patch in target.patches:
            lab = labs.get(patch.patch_id)
            if lab is None:
                continue
            if grayscale:
                Y = float(lab_to_xyz(lab, white_point=CIE_D50)[1])
                encoded = (np.clip(Y, 0.0, 1.0) ** (1.0 / (GRAY_GAMMA * gamma_error)),)
            else:
                linear = inverse @ lab_to_xyz(lab, white_point=CIE_D50)
                linear = np.clip(linear, 0.0, 1.0) * np.asarray(gain)
                encoded = np.clip(linear, 0.0, 1.0) ** (1.0 / (ADOBE_RGB_GAMMA * gamma_error))

            quad = np.array(grid.corners_relative(patch.row, patch.col, inset=1.0))
            homogeneous = np.column_stack([quad, np.ones(4)])
            projected = homogeneous @ homography.T
            projected = projected[:, :2] / projected[:, 2:3]
            cv2.fillPoly(canvas, [np.round(projected).astype(np.int32)], tuple(encoded))

        if ruler:
            _draw_sg_ruler(canvas, homography, grayscale=grayscale)

        data = np.clip(canvas, 0.0, 1.0)
        as_uint16 = np.round(data * 65535.0).astype(np.uint16)

        if grayscale:
            as_uint16 = as_uint16[..., 0]
            profile = icc if icc is not None else build_gray_icc(GRAY_GAMMA)
        else:
            profile = icc if icc is not None else adobe_rgb_icc()
        path.parent.mkdir(parents=True, exist_ok=True)
        # The header resolution the quad's own geometry implies, so a run's
        # reproduction scale accuracy compares two numbers that agree rather than
        # two that were never meant to. The quad is SG_EXTENT_MM wide by
        # construction, which is what the drawn ruler is spaced against.
        ppi = (corners[1][0] - corners[0][0]) / SG_EXTENT_MM * 25.4
        tifffile.imwrite(
            path,
            as_uint16,
            photometric="minisblack" if grayscale else "rgb",
            resolution=(ppi, ppi),
            resolutionunit="INCH",
            extratags=[(34675, "B", len(profile), profile, True)],
        )
        return path, corners


#: The width of the SG's fiducial quad, from the target definition. The synthetic
#: chart is rendered to this scale so its drawn ruler, its header resolution and
#: the definition's own extent all describe one chart rather than three.
SG_EXTENT_MM = 247.6


def _draw_sg_ruler(canvas: np.ndarray, homography: np.ndarray, *, grayscale: bool) -> None:
    """Draw the millimetre scale the SG prints below columns K-N.

    Light ticks on a dark ground, which is the way round this chart prints it and
    the reason `polarity` is left at `auto` in the definition. The comb is spaced
    at one millimetre in the quad's own units, so the pitch a run measures off it
    is the pitch the quad's geometry implies -- the fixture cannot flatter the
    measurement, because both come from the same number.
    """
    import cv2

    def project(points: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack([points, np.ones(len(points))])
        out = homogeneous @ homography.T
        return out[:, :2] / out[:, 2:3]

    ground = (0.16,) * (1 if grayscale else 3)
    ink = (0.92,) * (1 if grayscale else 3)

    band = np.array([[0.70, 1.004], [0.99, 1.004], [0.99, 1.068], [0.70, 1.068]])
    cv2.fillPoly(canvas, [np.round(project(band)).astype(np.int32)], ground)

    mm = 1.0 / SG_EXTENT_MM
    half = 0.00035  # half a tick's width, in quad units: about 2 px at this size
    # Ticks run past the declared band at both ends, as the printed ones do, so the
    # band is bounded by ink rather than by where the comb happens to stop.
    u = 0.7300 - 2 * mm
    while u <= 0.9600 + 2 * mm:
        tick = np.array([[u - half, 1.016], [u + half, 1.016],
                         [u + half, 1.052], [u - half, 1.052]])
        cv2.fillPoly(canvas, [np.round(project(tick)).astype(np.int32)], ink)
        u += mm


def _patch_sort_key(item: tuple[str, np.ndarray]) -> tuple[int, int]:
    patch_id = item[0]
    return (int(patch_id[1:]), COLS.index(patch_id[0]))


# -- LUT-based ICC profiles -------------------------------------------------
#
# Built so the LUT code path can be checked against the analytic one. The trick in
# `lut_icc_from_matrix` is to put the tone curve in the input curves and a *linear*
# function in the CLUT: multilinear interpolation reproduces a linear function
# exactly, so the LUT profile and the matrix profile must agree to the encoding's
# quantisation, not to interpolation error.

_LAB_LEGACY_DEN = 65280.0


def encode_xyz_u1f15(xyz: np.ndarray) -> np.ndarray:
    """PCS XYZ to normalized u1Fixed15 (0x8000 encodes 1.0)."""
    return np.clip(np.asarray(xyz, dtype=np.float64) / (65535.0 / 32768.0), 0.0, 1.0)


def encode_lab_legacy16(lab: np.ndarray) -> np.ndarray:
    """PCS Lab to normalized legacy 16-bit encoding (L* against 0xFF00)."""
    lab = np.asarray(lab, dtype=np.float64)
    L = lab[..., 0] * _LAB_LEGACY_DEN / (100.0 * 65535.0)
    a = (lab[..., 1] + 128.0) * _LAB_LEGACY_DEN / (255.0 * 65535.0)
    b = (lab[..., 2] + 128.0) * _LAB_LEGACY_DEN / (255.0 * 65535.0)
    return np.clip(np.stack([L, a, b], axis=-1), 0.0, 1.0)


def encode_lab_v4(lab: np.ndarray) -> np.ndarray:
    """PCS Lab to normalized ICC v4 encoding (L* against 0xFFFF)."""
    lab = np.asarray(lab, dtype=np.float64)
    L = lab[..., 0] / 100.0
    a = (lab[..., 1] + 128.0) / 255.0
    b = (lab[..., 2] + 128.0) / 255.0
    return np.clip(np.stack([L, a, b], axis=-1), 0.0, 1.0)


def _pad4(payload: bytes) -> bytes:
    return payload + b"\x00" * ((-len(payload)) % 4)


def _curv_tag(table: np.ndarray) -> bytes:
    values = np.clip(np.asarray(table, dtype=np.float64), 0.0, 1.0)
    raw = np.round(values * 65535.0).astype(">u2")
    return b"curv" + b"\x00" * 4 + struct.pack(">I", raw.size) + raw.tobytes()


def _identity_curve_tag() -> bytes:
    return b"curv" + b"\x00" * 4 + struct.pack(">I", 0)


def mft_tag(
    clut: np.ndarray,
    *,
    bits: int = 16,
    input_curves: list[np.ndarray] | None = None,
    output_curves: list[np.ndarray] | None = None,
    matrix: np.ndarray | None = None,
) -> bytes:
    """Build a lut8Type ('mft1') or lut16Type ('mft2') tag payload."""
    grid = clut.shape[0]
    assert clut.ndim == 4 and clut.shape[:3] == (grid, grid, grid)
    out_ch = clut.shape[3]

    entries = 256 if bits == 8 else 1024
    input_curves = input_curves or [np.linspace(0, 1, entries) for _ in range(3)]
    output_curves = output_curves or [np.linspace(0, 1, entries) for _ in range(out_ch)]
    matrix = np.eye(3) if matrix is None else matrix

    signature = b"mft1" if bits == 8 else b"mft2"
    head = signature + b"\x00" * 4 + struct.pack(">BBBB", 3, out_ch, grid, 0)
    head += b"".join(_s15f16(float(v)) for v in matrix.reshape(-1))

    if bits == 8:
        dtype, scale = ">u1", 255.0
        body = b""
    else:
        dtype, scale = ">u2", 65535.0
        body = struct.pack(">HH", len(input_curves[0]), len(output_curves[0]))

    def pack(values: np.ndarray) -> bytes:
        return np.round(np.clip(values, 0.0, 1.0) * scale).astype(dtype).tobytes()

    for curve in input_curves:
        body += pack(curve)
    body += pack(clut.reshape(-1))
    for curve in output_curves:
        body += pack(curve)

    return head + body


def mab_tag(
    clut: np.ndarray,
    *,
    a_curves: list[np.ndarray] | None = None,
    b_curves: list[np.ndarray] | None = None,
    m_curves: list[np.ndarray] | None = None,
    matrix: np.ndarray | None = None,
    precision: int = 2,
) -> bytes:
    """Build a lutAtoBType ('mAB ') tag payload.

    Processing order is A curves, CLUT, M curves, matrix, B curves.
    """
    grid = clut.shape[:-1]
    out_ch = clut.shape[-1]

    a_bytes = _pad4(b"".join(_curv_tag(c) for c in a_curves)) if a_curves else b""
    b_bytes = _pad4(
        b"".join(_curv_tag(c) for c in b_curves)
        if b_curves
        else b"".join(_identity_curve_tag() for _ in range(out_ch))
    )
    m_bytes = _pad4(b"".join(_curv_tag(c) for c in m_curves)) if m_curves else b""

    matrix_bytes = b""
    if matrix is not None:
        flat = list(np.asarray(matrix, dtype=np.float64).reshape(-1))
        if len(flat) == 9:
            flat += [0.0, 0.0, 0.0]
        matrix_bytes = _pad4(b"".join(_s15f16(float(v)) for v in flat))

    grid_bytes = bytes(list(grid) + [0] * (16 - len(grid)))
    dtype, scale = (">u1", 255.0) if precision == 1 else (">u2", 65535.0)
    data = np.round(np.clip(clut.reshape(-1), 0.0, 1.0) * scale).astype(dtype).tobytes()
    clut_bytes = _pad4(grid_bytes + struct.pack(">BBBB", precision, 0, 0, 0) + data)

    header_len = 32
    cursor = header_len
    offsets = {}
    for name, payload in (
        ("B", b_bytes), ("matrix", matrix_bytes), ("M", m_bytes),
        ("CLUT", clut_bytes), ("A", a_bytes),
    ):
        offsets[name] = cursor if payload else 0
        cursor += len(payload)

    head = (
        b"mAB "
        + b"\x00" * 4
        + struct.pack(">BBH", 3, out_ch, 0)
        + struct.pack(
            ">5I",
            offsets["B"], offsets["matrix"], offsets["M"], offsets["CLUT"], offsets["A"],
        )
    )
    return head + b_bytes + matrix_bytes + m_bytes + clut_bytes + a_bytes


def build_icc(
    tags: list[tuple[bytes, bytes]],
    *,
    profile_class: bytes = b"scnr",
    pcs: bytes = b"XYZ ",
    description: str = "Deltachrome test profile",
) -> bytes:
    """Assemble a structurally valid ICC profile from a list of (signature, payload)."""
    entries_tags: list[tuple[bytes, bytes]] = [
        (b"desc", _tag_desc(description)),
        (b"wtpt", _tag_xyz(np.array(CIE_D50))),
        *tags,
    ]

    table_size = 4 + 12 * len(entries_tags)
    offset = 128 + table_size
    entries: list[bytes] = []
    body = bytearray()
    for sig, payload in entries_tags:
        pad = (-len(payload)) % 4
        entries.append(struct.pack(">4sII", sig, offset, len(payload)))
        body += payload + b"\x00" * pad
        offset += len(payload) + pad

    header = bytearray(128)
    struct.pack_into(">I", header, 0, 128 + table_size + len(body))
    header[4:8] = b"none"
    struct.pack_into(">I", header, 8, 0x02400000)
    header[12:16] = profile_class
    header[16:20] = b"RGB "
    header[20:24] = pcs
    header[36:40] = b"acsp"
    struct.pack_into(">I", header, 64, 1)
    header[68:80] = _tag_xyz(np.array(CIE_D50))[8:20]

    return bytes(header) + struct.pack(">I", len(entries_tags)) + b"".join(entries) + bytes(body)


def linear_clut(matrix: np.ndarray, grid: int = 9) -> np.ndarray:
    """A CLUT holding XYZ = matrix @ linear_rgb, encoded as u1Fixed15.

    The stored function is linear in its inputs, and multilinear interpolation
    reproduces a linear function exactly -- so any disagreement with the analytic
    matrix profile is encoding quantisation, not interpolation error.
    """
    axis = np.linspace(0.0, 1.0, grid)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    linear = np.stack([r, g, b], axis=-1)
    return encode_xyz_u1f15(linear @ matrix.T)


def lut_icc_from_matrix(
    *,
    tag: bytes = b"A2B1",
    grid: int = 9,
    curve_points: int = 1024,
    tag_type: str = "mft2",
    profile_class: bytes = b"scnr",
    include_matrix_trc: bool = False,
    description: str = "Synthetic LUT Adobe RGB [Deltachrome test]",
    extra_tags: list[tuple[bytes, bytes]] | None = None,
) -> bytes:
    """A LUT profile that encodes exactly the Adobe RGB transform.

    The gamma curve lives in the input curves and the colorant matrix in the CLUT,
    so the LUT profile is analytically identical to the matrix profile and the two
    code paths can be compared directly.
    """
    matrix = adobe_rgb_matrix()
    device = np.linspace(0.0, 1.0, curve_points)
    trc = device**ADOBE_RGB_GAMMA
    clut = linear_clut(matrix, grid=grid)

    if tag_type == "mAB":
        payload = mab_tag(clut, a_curves=[trc] * 3)
    else:
        payload = mft_tag(clut, bits=8 if tag_type == "mft1" else 16, input_curves=[trc] * 3)

    tags: list[tuple[bytes, bytes]] = [(tag, payload)]
    if include_matrix_trc:
        tags += [
            (b"rXYZ", _tag_xyz(matrix[:, 0])),
            (b"gXYZ", _tag_xyz(matrix[:, 1])),
            (b"bXYZ", _tag_xyz(matrix[:, 2])),
            (b"rTRC", _tag_curve_gamma(ADOBE_RGB_GAMMA)),
            (b"gTRC", _tag_curve_gamma(ADOBE_RGB_GAMMA)),
            (b"bTRC", _tag_curve_gamma(ADOBE_RGB_GAMMA)),
        ]
    tags += extra_tags or []

    return build_icc(tags, profile_class=profile_class, pcs=b"XYZ ", description=description)


def constant_lut_icc(
    encoded: np.ndarray, *, tag_type: str = "mft2", pcs: bytes = b"Lab "
) -> bytes:
    """A profile whose CLUT holds one value at every corner.

    With every corner identical, interpolation is a no-op and the profile returns
    exactly that encoded value for any input -- which isolates the PCS decoding
    step so the Lab encodings can be tested without interpolation in the way.
    """
    clut = np.broadcast_to(np.asarray(encoded, dtype=np.float64), (2, 2, 2, 3)).copy()
    if tag_type == "mAB":
        payload = mab_tag(clut)
    else:
        payload = mft_tag(clut, bits=8 if tag_type == "mft1" else 16)
    return build_icc(
        [(b"A2B1", payload)], profile_class=b"scnr", pcs=pcs, description="constant LUT"
    )


def write_tiff_with_profile(
    path: Path, image_uint16: np.ndarray, icc: bytes, *, resolution: float | None = None
) -> Path:
    """``resolution`` writes a header ppi. Left out, the file claims none, which is
    itself a case worth testing -- a master with no stated resolution leaves every
    downstream consumer to guess."""
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if resolution is not None:
        extra = {"resolution": (resolution, resolution), "resolutionunit": 2}
    tifffile.imwrite(
        path,
        image_uint16,
        photometric="minisblack" if image_uint16.ndim == 2 else "rgb",
        extratags=[(34675, "B", len(icc), icc, True)],
        **extra,
    )
    return path


# -- JPEG 2000 containers ---------------------------------------------------
#
# JP2 is a preservation master format across the sector, and it stores its ICC
# profile in the container's `colr` box rather than anywhere Pillow surfaces. These
# helpers build real JP2 files carrying a real profile so the loader's box parsing
# is exercised against actual bytes rather than a mock.


def _jp2_boxes(data: bytes, start: int = 0, end: int | None = None):
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        length = struct.unpack_from(">I", data, pos)[0]
        btype = data[pos + 4 : pos + 8]
        if length == 0:
            length = end - pos
        yield btype, pos, length
        if length < 8:
            return
        pos += length


def write_jp2(
    path: Path,
    array: np.ndarray,
    *,
    icc: bytes | None = None,
    enum_cs: int | None = None,
    claim_bit_depth: int | None = None,
) -> Path:
    """Write a JP2 whose `colr` box carries ``icc`` (METH=2) or ``enum_cs`` (METH=1).

    ``claim_bit_depth`` rewrites the `ihdr` BPC field without changing the pixel
    data, which is how a file whose declared depth outruns what a decoder returns
    is simulated.

    **A uint16 array is encoded by OpenCV rather than Pillow**, because Pillow has
    no 16-bit RGB JPEG 2000 mode -- which is the whole reason the loader stopped
    decoding through it. Both writers emit the same box structure, so the `colr`
    surgery below is unchanged either way.
    """
    import io

    if array.dtype == np.uint16:
        import cv2

        # Explicitly lossless. OpenCV's JP2 encoder defaults to lossy, which on a
        # 16-bit fixture moves pixels by thousands of counts -- so a test asserting
        # the loader reads the file back exactly would fail on the writer.
        ok, encoded = cv2.imencode(
            ".jp2",
            array[:, :, ::-1] if array.ndim == 3 else array,
            [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 1000],
        )
        assert ok, "OpenCV declined to encode the fixture"
        data = bytearray(encoded.tobytes())
    else:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG2000", irreversible=False)
        data = bytearray(buffer.getvalue())

    header = next((p, n) for t, p, n in _jp2_boxes(data) if t == b"jp2h")
    jp2h_pos, jp2h_len = header

    if icc is not None:
        colr = struct.pack(">BbB", 2, 0, 0) + icc
    elif enum_cs is not None:
        colr = struct.pack(">BbBI", 1, 0, 0, enum_cs)
    else:
        colr = None

    if colr is not None:
        box = struct.pack(">I", 8 + len(colr)) + b"colr" + colr
        old = next(
            (p, n) for t, p, n in _jp2_boxes(data, jp2h_pos + 8, jp2h_pos + jp2h_len)
            if t == b"colr"
        )
        old_pos, old_len = old
        data[old_pos : old_pos + old_len] = box
        struct.pack_into(">I", data, jp2h_pos, jp2h_len - old_len + len(box))

    if claim_bit_depth is not None:
        ihdr = next(
            (p, n) for t, p, n in _jp2_boxes(data, jp2h_pos + 8, jp2h_pos + jp2h_len)
            if t == b"ihdr"
        )
        # ihdr after its 8-byte header: HEIGHT(4) WIDTH(4) NC(2) BPC(1)
        data[ihdr[0] + 18] = claim_bit_depth - 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(data))
    return path


def render_flat_field(
    path: Path,
    *,
    size: tuple[int, int] = (900, 700),
    lightness: tuple[float, float] = (95.0, 95.0),
    b_star: tuple[float, float] = (0.0, 0.0),
    level: float | None = None,
) -> Path:
    """A frame-filling white sheet with a known left-to-right gradient.

    ``lightness`` and ``b_star`` give the value at the left and right edges, and
    everything between is a linear ramp -- so the spread the metric should recover
    is exactly the difference between the two ends, and where the extremes are is
    known too. A real illumination fall-off is radial rather than linear, but a
    ramp makes the expected answer arithmetic rather than a second model.

    ``level`` overrides every pixel with one encoded value, for the clipping case.
    """
    import tifffile

    width, height = size
    if level is not None:
        data = np.full((height, width, 3), float(level))
    else:
        u = np.linspace(0.0, 1.0, width)[None, :, None]
        lab = np.concatenate(
            [
                lightness[0] + u * (lightness[1] - lightness[0]),
                np.zeros_like(u),
                b_star[0] + u * (b_star[1] - b_star[0]),
            ],
            axis=-1,
        )
        linear = lab_to_xyz(np.broadcast_to(lab, (1, width, 3)), white_point=CIE_D50)
        linear = linear @ np.linalg.inv(adobe_rgb_matrix()).T
        encoded = np.clip(linear, 0.0, 1.0) ** (1.0 / ADOBE_RGB_GAMMA)
        data = np.broadcast_to(encoded, (height, width, 3))

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = adobe_rgb_icc()
    tifffile.imwrite(
        path,
        np.round(np.clip(data, 0.0, 1.0) * 65535.0).astype(np.uint16),
        photometric="rgb",
        extratags=[(34675, "B", len(profile), profile, True)],
    )
    return path
