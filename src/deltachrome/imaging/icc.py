"""Exact ICC profile evaluation, in float, for both profile structures.

Why hand-parse instead of leaning on a binding?

The only widely-available lcms binding in Python is Pillow's ``ImageCms``, and its
Lab output mode is 8 bits per channel. Quantising L*a*b* to 8 bits injects on the
order of 0.4 Delta E of error before any measurement happens -- more than 10% of a
FADGI 3-star 3.5 Delta E budget, spent on nothing. For a tool whose entire job is
to report Delta E to two decimals, that is not an acceptable floor.

ICC describes a device transform in one of two structurally different ways, and
Deltachrome evaluates both at full float precision:

**Matrix/TRC.** The transform stated analytically: three tone curves and a 3x3
colorant matrix. ``device RGB --TRC--> linear RGB --matrix--> XYZ``. Every RGB
working space -- sRGB, Adobe RGB (1998), ProPhoto, ECI RGB v2, Rec.2020,
Display P3 -- is described this way, and for those the matrix *is* the definition.

**LUT-based (``A2B0``/``A2B1``/``A2B2``).** The transform sampled into a
multidimensional lookup table, wrapped in per-channel curves. This exists because a
real device's response is not separable into per-channel curves plus a linear
matrix -- channel crosstalk and non-separable colorant interaction are exactly what
a scanner or camera profile has to capture. Three tag encodings appear in the wild
(``mft1``, ``mft2``, ``mAB ``) and all three are handled here.

Two traps in the LUT path are worth naming, because getting either wrong produces
results that look entirely plausible:

* **Legacy vs. v4 PCS Lab encoding.** ``lut16Type`` encodes L* against 0xFF00,
  while ``lutAtoBType`` encodes it against 0xFFFF. Using one decoder for both puts
  L* out by a factor of 1.0039 -- about 0.38 L* at the highlight end of a tonal
  ramp, comfortably enough to flip a FADGI tone response result.
* **Rendering intent.** ``A2B0`` is perceptual, which includes gamut mapping. For
  measurement we want ``A2B1``, media-relative colorimetric, and fall back only if
  it is absent. Measuring through a perceptual table would report the profile's
  artistic intent as if it were the device's response.
"""

from __future__ import annotations

import itertools
import struct
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from deltachrome.errors import UnsupportedICCProfileError

# The ICC profile connection space white point, D50, as encoded in the spec.
ICC_PCS_D50 = (0.96420, 1.00000, 0.82491)
# CIE D50 (x=0.34567, y=0.35850), which is what a spectrophotometer reports Lab
# against. The two differ by ~0.04% in Z -- about 0.02 Delta E. We use CIE D50 for
# both image and reference so the two sides of the comparison share a white point.
CIE_D50 = (0.964212, 1.00000, 0.825188)

_S15F16 = 65536.0
_U8F8 = 256.0

#: A2B tag preference. A2B1 is media-relative colorimetric -- the device's actual
#: response. A2B0 is perceptual and carries gamut mapping, which would be measured
#: as if it were device behavior. A2B2 (saturation) is a last resort.
A2B_PREFERENCE = ("A2B1", "A2B0", "A2B2")

#: Profile classes whose device response is genuinely non-separable, so a LUT is
#: the authoritative description and any matrix/TRC tags are a fallback
#: approximation left for simple CMMs.
DEVICE_CLASSES = frozenset({"scnr", "prtr", "cmra"})

#: Samples used to read a LUT profile's neutral-axis response, when a per-channel
#: linearization is needed and the profile carries no separable per-channel curve.
#: See :meth:`ICCProfile.channel_linearizer`.
NEUTRAL_AXIS_SAMPLES = 256

#: Device colour spaces this reader evaluates, and how many channels each carries.
#:
#: Grayscale is here because a monochrome capture is not an oddity in this domain --
#: it is what a microfilm scanner produces, and Gray Gamma 2.2 is a profile FADGI's
#: own microform samples ship with. What it is *not* is a colour capture with the
#: chroma set to zero: it carries one colour record where an RGB file carries three,
#: which is why the channel count travels with the profile rather than being assumed
#: at every call site. CMYK and >4-channel spaces stay refused: they need a LUT
#: reader with more than three inputs and a pixel pipeline that does not exist yet.
DEVICE_CHANNELS = {"RGB ": 3, "GRAY": 1}


@dataclass(frozen=True)
class ToneCurve:
    """An ICC tone reproduction curve, evaluable in float.

    Wraps a ``curv`` tag (identity, pure gamma, or a sampled table), a ``para`` tag
    (one of the seven parametric forms in ICC.1:2010), or a raw sampled table taken
    from inside a ``mft1``/``mft2`` pipeline.
    """

    kind: str
    gamma: float | None = None
    table: np.ndarray | None = None
    params: tuple[float, ...] = ()
    para_type: int | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Map device values in [0, 1] to linear values in [0, 1]."""
        x = np.asarray(x, dtype=np.float64)
        if self.kind == "identity":
            return x
        if self.kind == "gamma":
            assert self.gamma is not None
            return np.power(np.clip(x, 0.0, None), self.gamma)
        if self.kind == "table":
            # curveType semantics: a single entry means gamma, not a table.
            assert self.table is not None
            if self.table.size == 1:
                return np.power(np.clip(x, 0.0, None), float(self.table[0]))
            return self._interpolate(x)
        if self.kind == "sampled":
            # Pipeline curve semantics: always a table, whatever its length.
            return self._interpolate(x)
        if self.kind == "parametric":
            return self._parametric(x)
        raise UnsupportedICCProfileError(f"Unknown tone curve kind {self.kind!r}")

    def _interpolate(self, x: np.ndarray) -> np.ndarray:
        assert self.table is not None
        n = self.table.size
        if n == 0:
            return x
        if n == 1:
            return np.full_like(x, float(self.table[0]))
        grid = np.linspace(0.0, 1.0, n)
        return np.interp(np.clip(x, 0.0, 1.0), grid, self.table)

    def _parametric(self, x: np.ndarray) -> np.ndarray:
        p = self.params
        t = self.para_type
        if t == 0:
            (g,) = p
            return np.power(np.clip(x, 0.0, None), g)
        if t == 1:
            g, a, b = p
            return np.where(x >= -b / a, np.power(np.clip(a * x + b, 0.0, None), g), 0.0)
        if t == 2:
            g, a, b, c = p
            return np.where(x >= -b / a, np.power(np.clip(a * x + b, 0.0, None), g) + c, c)
        if t == 3:
            g, a, b, c, d = p
            return np.where(x >= d, np.power(np.clip(a * x + b, 0.0, None), g), c * x)
        if t == 4:
            g, a, b, c, d, e, f = p
            return np.where(x >= d, np.power(np.clip(a * x + b, 0.0, None), g) + e, c * x + f)
        raise UnsupportedICCProfileError(
            f"Parametric curve function type {t} is not defined in ICC.1:2010."
        )

    def describe(self) -> str:
        if self.kind == "identity":
            return "identity"
        if self.kind == "gamma":
            return f"gamma {self.gamma:.4f}"
        if self.kind in ("table", "sampled"):
            assert self.table is not None
            if self.kind == "table" and self.table.size == 1:
                return f"gamma {float(self.table[0]):.4f}"
            return f"sampled table, {self.table.size} points"
        return f"parametric type {self.para_type} {tuple(round(v, 6) for v in self.params)}"


IDENTITY_CURVE = ToneCurve(kind="identity")


@dataclass(frozen=True)
class CLUT:
    """A multidimensional color lookup table.

    ``table`` has shape ``grid + (output_channels,)`` and holds values in [0, 1].

    **Three inputs are interpolated tetrahedrally**, which is how the profile's
    author evaluated the table and so the only reading of it that is the
    author's. Argyll chooses simplex (tetrahedral) interpolation for any RGB-input
    table (``icc/icc_xf.c``, ``icmPeClut_choose_alg``), and lcms does the same by
    default for three inputs. This used to be trilinear, under a docstring saying
    lcms was trilinear too, and on a camera table built by ``colprof`` the two
    disagreed by up to 5.45 ΔE2000 near black: trilinear weighs
    all eight corners of a cell, and in the first cell of a device-RGB table the
    six off-diagonal corners are strongly coloured, so a neutral input came out
    tinted. A tetrahedron always has the cell's black and white corners as two of
    its four, and a neutral input weighs almost nothing else. Any other number of
    inputs is multilinear.
    """

    grid: tuple[int, ...]
    table: np.ndarray

    @property
    def input_channels(self) -> int:
        return len(self.grid)

    @property
    def output_channels(self) -> int:
        return int(self.table.shape[-1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Interpolate. ``x`` has shape (..., input_channels), values in [0, 1]."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.input_channels:
            raise UnsupportedICCProfileError(
                f"CLUT expects {self.input_channels} input channels, got {x.shape[-1]}."
            )
        shape = x.shape[:-1]
        flat = np.clip(x.reshape(-1, self.input_channels), 0.0, 1.0)

        dims = np.array(self.grid, dtype=np.int64)
        coords = flat * (dims - 1)
        base = np.floor(coords).astype(np.int64)
        # A degenerate axis (one grid point) has no interval to interpolate across.
        upper = np.maximum(dims - 2, 0)
        base = np.clip(base, 0, upper)
        frac = np.where(dims > 1, coords - base, 0.0)

        if self.input_channels == 3 and np.all(dims > 1):
            return self._tetrahedral(base, frac).reshape(shape + (self.output_channels,))

        out = np.zeros((flat.shape[0], self.output_channels), dtype=np.float64)
        for corner in itertools.product((0, 1), repeat=self.input_channels):
            weight = np.ones(flat.shape[0], dtype=np.float64)
            index = []
            for axis, step in enumerate(corner):
                if step:
                    weight = weight * frac[:, axis]
                    index.append(np.minimum(base[:, axis] + 1, dims[axis] - 1))
                else:
                    weight = weight * (1.0 - frac[:, axis])
                    index.append(base[:, axis])
            if not np.any(weight):
                continue
            out += weight[:, None] * self.table[tuple(index)]

        return out.reshape(shape + (self.output_channels,))

    def _tetrahedral(self, base: np.ndarray, frac: np.ndarray) -> np.ndarray:
        """Interpolate inside the tetrahedron of the cell that holds each point.

        The cube is cut along its black-to-white diagonal into six tetrahedra, one
        per ordering of the three fractional coordinates. Walking from the cell's
        base corner, step along the axis with the largest fraction, then the next,
        then the last: the four corners visited bound the point, and its weights
        are the differences between successive sorted fractions.
        """
        rows = np.arange(len(frac))
        order = np.argsort(-frac, axis=1, kind="stable")
        steps = np.take_along_axis(frac, order, axis=1)
        corner = base.copy()
        out = (1.0 - steps[:, 0])[:, None] * self.table[tuple(corner.T)]
        for k in range(3):
            corner[rows, order[:, k]] += 1
            weight = steps[:, k] - (steps[:, k + 1] if k < 2 else 0.0)
            out += weight[:, None] * self.table[tuple(corner.T)]
        return out


@dataclass(frozen=True)
class LutPipeline:
    """An ``A2B`` transform chain: A curves, CLUT, M curves, matrix, B curves.

    ``mft1``/``mft2`` use only the input-curve, CLUT and output-curve stages;
    ``mAB `` may use all five. Stages that a profile omits are ``None`` and skipped.
    """

    tag: str
    tag_type: str
    pcs: str
    pcs_encoding: str
    input_curves: tuple[ToneCurve, ...] | None = None
    clut: CLUT | None = None
    m_curves: tuple[ToneCurve, ...] | None = None
    matrix: np.ndarray | None = None
    offset: np.ndarray | None = None
    output_curves: tuple[ToneCurve, ...] | None = None

    def to_pcs_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """Evaluate device RGB in [0, 1] to PCS XYZ (D50-adapted)."""
        x = np.asarray(rgb, dtype=np.float64)
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        if self.input_curves is not None:
            x = _apply_curves(x, self.input_curves, "A/input")
        if self.clut is not None:
            x = self.clut(x)
        if self.m_curves is not None:
            x = _apply_curves(x, self.m_curves, "M")
        if self.matrix is not None:
            x = x @ self.matrix.T
            if self.offset is not None:
                x = x + self.offset
        if self.output_curves is not None:
            x = _apply_curves(x, self.output_curves, "B/output")

        return _decode_pcs(x, self.pcs_encoding).reshape(shape + (3,))

    def describe(self) -> str:
        stages = []
        if self.input_curves:
            stages.append(f"{len(self.input_curves)} input curves")
        if self.clut:
            stages.append(f"CLUT {'x'.join(str(g) for g in self.clut.grid)}")
        if self.m_curves:
            stages.append(f"{len(self.m_curves)} M curves")
        if self.matrix is not None:
            stages.append("3x3 matrix")
        if self.output_curves:
            stages.append(f"{len(self.output_curves)} output curves")
        return (
            f"{self.tag} ({self.tag_type.strip()}) -> PCS {self.pcs} "
            f"[{self.pcs_encoding}]: " + " -> ".join(stages)
        )


@dataclass(frozen=True)
class ICCProfile:
    """A parsed profile able to convert device RGB to PCS XYZ exactly."""

    description: str
    profile_class: str
    color_space: str
    pcs: str
    white_point: tuple[float, float, float]
    raw_size: int
    transform_kind: str  # "matrix_trc" or "lut"
    matrix: np.ndarray | None = None
    trc: tuple[ToneCurve, ToneCurve, ToneCurve] | None = None
    pipeline: LutPipeline | None = None
    available_transforms: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def channels(self) -> int:
        """Device channels this profile transforms -- 3 for RGB, 1 for grayscale.

        Read from the profile's own colour space rather than inferred from the
        pixels, so a grayscale file carrying an RGB profile (or the reverse) is a
        mismatch the loader can refuse instead of a shape error somewhere deeper.
        """
        return DEVICE_CHANNELS.get(self.color_space.ljust(4), 3)

    @property
    def is_grayscale(self) -> bool:
        return self.channels == 1

    def to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """Convert device values in [0, 1] to PCS XYZ (D50-adapted).

        ``rgb`` may be any shape whose trailing axis matches :attr:`channels`; the
        result carries a trailing axis of 3 either way, because the PCS is XYZ
        whatever the device space was.

        **Grayscale goes through a different transform, not a degenerate one.** A
        monochrome profile states a single tone curve and no colorants: ICC defines
        the result as ``Y = kTRC(g)`` scaled onto the PCS illuminant, so the output
        is neutral by construction. Feeding a grey through the RGB path by
        replicating it into three channels would be the same arithmetic only when
        the three colorants happen to sum to the illuminant, which is a property of
        the profile rather than a rule -- and it would also invent two colour
        records the file does not have.
        """
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.shape[-1] != self.channels:
            raise ValueError(
                f"Expected trailing axis of {self.channels} for a "
                f"{self.color_space.strip()} profile, got shape {rgb.shape}"
            )

        if self.transform_kind == "matrix_trc":
            assert self.matrix is not None and self.trc is not None
            linear = np.stack(
                [self.trc[i](rgb[..., i]) for i in range(self.channels)], axis=-1
            )
            return linear @ self.matrix.T

        assert self.pipeline is not None
        return self.pipeline.to_pcs_xyz(rgb)

    def channel_linearizer(self) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        """A per-channel map from encoded value to a signal linear in luminance.

        Returned with a short description of the construction used, because which
        one applies is part of the measurement.

        The constraint that makes this a separate method from :meth:`to_xyz` is
        that it must **not mix the channels**. Anything measured per color record
        -- channel registration above all -- is destroyed by a 3x3 to XYZ: each
        output would be a blend of all three records, so a displacement between
        two of them would be averaged away into all three. Only a per-channel
        monotone curve is admissible.

        ``matrix_trc`` profiles carry exactly that, one tone curve per channel. LUT
        profiles carry no separable per-channel curve, so the profile's neutral-axis
        response is used for all three records instead. That is not a general
        substitute, but it is the right curve for the features this is used on: a
        slanted edge cut into a neutral patch travels along the neutral axis in
        every record.
        """
        if self.transform_kind == "matrix_trc" and self.trc is not None:
            trc = self.trc
            count = self.channels

            def per_channel(rgb: np.ndarray) -> np.ndarray:
                rgb = np.asarray(rgb, dtype=np.float64)
                return np.stack([trc[i](rgb[..., i]) for i in range(count)], axis=-1)

            return per_channel, "per-channel TRC from the embedded profile"

        samples = np.linspace(0.0, 1.0, NEUTRAL_AXIS_SAMPLES)
        response = self.to_xyz(np.stack([samples] * self.channels, axis=-1))[..., 1]
        if not np.all(np.diff(response) > 0):
            return (
                lambda rgb: np.asarray(rgb, dtype=np.float64),
                "none -- the profile's neutral axis is not monotonic in luminance",
            )

        def neutral_axis(rgb: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(rgb, dtype=np.float64), samples, response)

        return neutral_axis, "neutral-axis response of the embedded LUT transform"

    def summary(self) -> str:
        lines = [
            f"{self.description} [{self.profile_class}/{self.color_space}->{self.pcs}]",
            f"  transform: {self.transform_kind}",
        ]
        if self.transform_kind == "matrix_trc":
            assert self.trc is not None
            names = "RGB" if self.channels == 3 else "K"
            lines += [
                f"  {names[i]} TRC: {self.trc[i].describe()}"
                for i in range(self.channels)
            ]
        else:
            assert self.pipeline is not None
            lines.append(f"  {self.pipeline.describe()}")
        if self.available_transforms:
            lines.append(f"  transforms present: {', '.join(self.available_transforms)}")
        lines.append(
            "  white point (PCS XYZ): "
            f"({self.white_point[0]:.5f}, {self.white_point[1]:.5f}, {self.white_point[2]:.5f})"
        )
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def parse_icc(data: bytes, *, prefer: str | None = None) -> ICCProfile:
    """Parse an ICC profile blob.

    ``prefer`` forces a transform structure -- ``"matrix_trc"`` or ``"lut"`` -- for
    the unusual profile that carries both and disagrees with the default policy.
    """
    if len(data) < 132:
        raise UnsupportedICCProfileError(
            f"ICC blob is only {len(data)} bytes; too short to be a profile."
        )

    declared_size = struct.unpack_from(">I", data, 0)[0]
    profile_class = _sig(data, 12).strip()
    color_space = _sig(data, 16)
    pcs = _sig(data, 20)

    if color_space not in DEVICE_CHANNELS:
        raise UnsupportedICCProfileError(
            f"Image profile color space is {color_space.strip()!r}. Deltachrome "
            "handles RGB and grayscale profiles. CMYK and multi-channel (>3) are on "
            "the roadmap and will need this reader extended."
        )
    channels = DEVICE_CHANNELS[color_space]

    tags = _read_tag_table(data)
    notes: list[str] = []

    matrix_tags = (
        ("rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC") if channels == 3 else ("kTRC",)
    )
    has_matrix = set(matrix_tags) <= tags.keys()
    a2b_tag = next((t for t in A2B_PREFERENCE if t in tags), None)

    available = tuple(
        (["matrix/TRC"] if has_matrix else []) + [t for t in A2B_PREFERENCE if t in tags]
    )

    kind = _choose_transform(
        prefer=prefer,
        has_matrix=has_matrix,
        a2b_tag=a2b_tag,
        profile_class=profile_class,
        tags=tags,
        notes=notes,
    )

    white_point = tuple(_read_xyz(data, tags["wtpt"])) if "wtpt" in tags else ICC_PCS_D50
    description = _read_description(data, tags) or "(unnamed profile)"

    common = {
        "description": description,
        "profile_class": profile_class,
        "color_space": color_space.strip(),
        "pcs": pcs.strip(),
        "white_point": white_point,
        "raw_size": declared_size,
        "available_transforms": available,
    }

    if kind == "matrix_trc":
        if channels == 1:
            # ICC clause 8.3: a monochrome profile's PCS value is the tone curve's
            # output scaled onto the PCS illuminant. Writing that as a 3x1 colorant
            # matrix keeps one evaluation path for both spaces -- the grey is the
            # single "colorant" and D50 is its tristimulus.
            matrix = np.asarray(ICC_PCS_D50, dtype=np.float64).reshape(3, 1)
            trc = (_read_curve(data, tags["kTRC"], "kTRC"),)
            notes.append(
                "Grayscale capture: color and registration not measured."
            )
        else:
            matrix = np.column_stack(
                [_read_xyz(data, tags[t]) for t in ("rXYZ", "gXYZ", "bXYZ")]
            )
            trc = tuple(_read_curve(data, tags[t], t) for t in ("rTRC", "gTRC", "bTRC"))
        return ICCProfile(
            transform_kind="matrix_trc",
            matrix=matrix,
            trc=trc,  # type: ignore[arg-type]
            notes=tuple(notes),
            **common,  # type: ignore[arg-type]
        )

    assert a2b_tag is not None
    pipeline = _read_a2b(data, tags[a2b_tag], a2b_tag, pcs)
    if a2b_tag == "A2B0" and "A2B1" not in tags:
        notes.append(
            "Profile lacks a colorimetric table; perceptual table used."
        )
    return ICCProfile(
        transform_kind="lut",
        pipeline=pipeline,
        notes=tuple(notes),
        **common,  # type: ignore[arg-type]
    )


def _choose_transform(
    *,
    prefer: str | None,
    has_matrix: bool,
    a2b_tag: str | None,
    profile_class: str,
    tags: dict[str, tuple[int, int]],
    notes: list[str],
) -> str:
    """Decide which transform structure to evaluate.

    Policy, and the reasoning behind it:

    * A device profile (``scnr``, ``prtr``, ``cmra``) exists because the device's
      response is not separable. Its LUT is the authoritative description; any
      matrix/TRC tags it also carries are a coarse fallback for simple CMMs.
    * A working-space or display profile (``mntr``, ``spac``) is *defined* by its
      matrix and curves. A LUT there is a sampled approximation of an exact
      analytic transform, so using it would only add interpolation error.
    """
    if prefer is not None:
        if prefer not in ("matrix_trc", "lut"):
            raise ValueError(f"prefer must be 'matrix_trc' or 'lut', got {prefer!r}")
        if prefer == "matrix_trc" and not has_matrix:
            raise UnsupportedICCProfileError(
                "matrix/TRC evaluation was requested but the profile has no complete "
                f"colorant + TRC tag set. Tags present: {sorted(tags)}"
            )
        if prefer == "lut" and a2b_tag is None:
            raise UnsupportedICCProfileError(
                "LUT evaluation was requested but the profile has no A2B tag. "
                f"Tags present: {sorted(tags)}"
            )
        return prefer

    if profile_class in DEVICE_CLASSES and a2b_tag is not None:
        return "lut"

    if has_matrix:
        return "matrix_trc"

    if a2b_tag is not None:
        return "lut"

    raise UnsupportedICCProfileError(
        "Profile has neither a complete matrix/TRC tag set (rXYZ/gXYZ/bXYZ + "
        f"rTRC/gTRC/bTRC) nor an A2B tag; tags present: {sorted(tags)}"
    )


# -- A2B pipeline reading ---------------------------------------------------


def _read_a2b(data: bytes, loc: tuple[int, int], tag: str, pcs: str) -> LutPipeline:
    offset, size = loc
    tag_type = _sig(data, offset)
    pcs_name = pcs.strip()
    if pcs_name not in ("XYZ", "Lab"):
        raise UnsupportedICCProfileError(
            f"{tag}: profile connection space is {pcs_name!r}; expected XYZ or Lab."
        )

    if tag_type == "mft1":
        return _read_mft(data, offset, size, tag, tag_type, pcs_name, bits=8)
    if tag_type == "mft2":
        return _read_mft(data, offset, size, tag, tag_type, pcs_name, bits=16)
    if tag_type == "mAB ":
        return _read_mab(data, offset, size, tag, pcs_name)
    raise UnsupportedICCProfileError(
        f"{tag}: unsupported LUT tag type {tag_type!r}. Deltachrome reads "
        "lut8Type ('mft1'), lut16Type ('mft2') and lutAtoBType ('mAB ')."
    )


def _read_mft(
    data: bytes, offset: int, size: int, tag: str, tag_type: str, pcs: str, *, bits: int
) -> LutPipeline:
    """Read a ``lut8Type``/``lut16Type`` tag.

    Structure: an optional 3x3 matrix (identity unless the *input* is PCSXYZ, which
    it never is for a device profile), per-input-channel curves, a CLUT, and
    per-output-channel curves.
    """
    in_ch, out_ch, grid_points = struct.unpack_from(">BBB", data, offset + 8)
    if in_ch != 3:
        raise UnsupportedICCProfileError(
            f"{tag}: {in_ch} input channels. Deltachrome v1 handles 3-channel RGB input."
        )
    if out_ch != 3:
        raise UnsupportedICCProfileError(
            f"{tag}: {out_ch} output channels; a PCS transform must produce 3."
        )
    if grid_points < 2:
        raise UnsupportedICCProfileError(f"{tag}: CLUT declares {grid_points} grid points.")

    matrix = np.array(struct.unpack_from(">9i", data, offset + 12), dtype=np.float64)
    matrix = (matrix / _S15F16).reshape(3, 3)
    if not np.allclose(matrix, np.eye(3), atol=1e-4):
        # Per ICC.1:2010 this matrix applies only when the input space is PCSXYZ.
        # A device RGB profile carrying a non-identity matrix here is malformed, and
        # applying it anyway would silently distort every value.
        raise UnsupportedICCProfileError(
            f"{tag}: the tag's 3x3 matrix is not the identity, but the input space is "
            "device RGB. Per ICC.1:2010 that matrix applies only to PCSXYZ input, so "
            "this profile is malformed."
        )

    if bits == 8:
        cursor = offset + 48
        in_entries = out_entries = 256
        dtype = np.uint8
        scale = 255.0
    else:
        in_entries, out_entries = struct.unpack_from(">HH", data, offset + 48)
        cursor = offset + 52
        dtype = np.dtype(">u2")
        scale = 65535.0
        if in_entries < 2 or out_entries < 2:
            raise UnsupportedICCProfileError(
                f"{tag}: curve tables declare {in_entries}/{out_entries} entries; "
                "ICC requires at least 2."
            )

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        item = np.dtype(dtype).itemsize
        need = count * item
        if cursor + need > offset + size:
            raise UnsupportedICCProfileError(
                f"{tag}: tag is {size} bytes but its declared tables need more."
            )
        values = np.frombuffer(data, dtype=dtype, count=count, offset=cursor)
        cursor += need
        return values.astype(np.float64) / scale

    input_curves = tuple(
        ToneCurve(kind="sampled", table=take(in_entries)) for _ in range(in_ch)
    )
    clut_values = take(grid_points**in_ch * out_ch)
    output_curves = tuple(
        ToneCurve(kind="sampled", table=take(out_entries)) for _ in range(out_ch)
    )

    clut = CLUT(
        grid=(grid_points,) * in_ch,
        table=clut_values.reshape((grid_points,) * in_ch + (out_ch,)),
    )

    return LutPipeline(
        tag=tag,
        tag_type=tag_type,
        pcs=pcs,
        pcs_encoding=_mft_pcs_encoding(pcs, bits, tag),
        input_curves=input_curves,
        clut=clut,
        output_curves=output_curves,
    )


def _mft_pcs_encoding(pcs: str, bits: int, tag: str) -> str:
    if pcs == "Lab":
        # lut16Type uses the *legacy* 16-bit Lab encoding (L* against 0xFF00), not
        # the v4 encoding. Confusing the two shifts L* by a factor of 1.0039.
        return "lab_legacy16" if bits == 16 else "lab_8"
    if bits == 8:
        raise UnsupportedICCProfileError(
            f"{tag}: lut8Type with a PCSXYZ connection space has no defined encoding "
            "in ICC.1:2010, and 8-bit XYZ would be far too coarse to measure through."
        )
    return "xyz_u1f15"


def _read_mab(data: bytes, offset: int, size: int, tag: str, pcs: str) -> LutPipeline:
    """Read a ``lutAtoBType`` tag.

    Processing order for A-to-B is: A curves, CLUT, M curves, matrix, B curves. Any
    element whose offset is zero is absent and skipped.
    """
    in_ch, out_ch = struct.unpack_from(">BB", data, offset + 8)
    if in_ch != 3:
        raise UnsupportedICCProfileError(
            f"{tag}: {in_ch} input channels. Deltachrome v1 handles 3-channel RGB input."
        )
    if out_ch != 3:
        raise UnsupportedICCProfileError(
            f"{tag}: {out_ch} output channels; a PCS transform must produce 3."
        )

    off_b, off_matrix, off_m, off_clut, off_a = struct.unpack_from(">5I", data, offset + 12)

    a_curves = _read_curve_set(data, offset + off_a, in_ch, tag, "A") if off_a else None
    m_curves = _read_curve_set(data, offset + off_m, out_ch, tag, "M") if off_m else None
    b_curves = _read_curve_set(data, offset + off_b, out_ch, tag, "B") if off_b else None

    clut = _read_mab_clut(data, offset + off_clut, in_ch, out_ch, tag) if off_clut else None

    matrix = offset_vector = None
    if off_matrix:
        raw = struct.unpack_from(">12i", data, offset + off_matrix)
        values = np.array(raw, dtype=np.float64) / _S15F16
        matrix = values[:9].reshape(3, 3)
        offset_vector = values[9:]

    if a_curves is not None and clut is None:
        raise UnsupportedICCProfileError(
            f"{tag}: A curves are present without a CLUT, which ICC.1:2010 forbids."
        )
    if m_curves is not None and matrix is None:
        raise UnsupportedICCProfileError(
            f"{tag}: M curves are present without a matrix, which ICC.1:2010 forbids."
        )
    if b_curves is None:
        raise UnsupportedICCProfileError(
            f"{tag}: B curves are required in a lutAtoBType tag and are absent."
        )

    return LutPipeline(
        tag=tag,
        tag_type="mAB ",
        pcs=pcs,
        # lutAtoBType always uses the ICC v4 PCS encodings.
        pcs_encoding="lab_v4_16" if pcs == "Lab" else "xyz_u1f15",
        input_curves=a_curves,
        clut=clut,
        m_curves=m_curves,
        matrix=matrix,
        offset=offset_vector,
        output_curves=b_curves,
    )


def _read_curve_set(
    data: bytes, offset: int, count: int, tag: str, label: str
) -> tuple[ToneCurve, ...]:
    """Read ``count`` consecutive curve tags, each padded to a 4-byte boundary."""
    curves: list[ToneCurve] = []
    cursor = offset
    for i in range(count):
        kind = _sig(data, cursor)
        if kind == "curv":
            entries = struct.unpack_from(">I", data, cursor + 8)[0]
            length = 12 + entries * 2
        elif kind == "para":
            func = struct.unpack_from(">H", data, cursor + 8)[0]
            counts = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
            if func not in counts:
                raise UnsupportedICCProfileError(
                    f"{tag}: {label} curve {i} uses parametric function {func}, "
                    "which is not defined in ICC.1:2010."
                )
            length = 12 + counts[func] * 4
        else:
            raise UnsupportedICCProfileError(
                f"{tag}: {label} curve {i} has type {kind!r}; expected 'curv' or 'para'."
            )
        curves.append(_read_curve(data, (cursor, length), f"{tag} {label}{i}"))
        cursor += length + ((-length) % 4)
    return tuple(curves)


def _read_mab_clut(
    data: bytes, offset: int, in_ch: int, out_ch: int, tag: str
) -> CLUT:
    grid = tuple(int(g) for g in data[offset : offset + in_ch])
    if any(g < 2 for g in grid):
        raise UnsupportedICCProfileError(f"{tag}: CLUT grid sizes {grid} include a value < 2.")
    precision = data[offset + 16]
    if precision == 1:
        dtype, scale = np.uint8, 255.0
    elif precision == 2:
        dtype, scale = np.dtype(">u2"), 65535.0
    else:
        raise UnsupportedICCProfileError(
            f"{tag}: CLUT precision byte is {precision}; ICC defines 1 (8-bit) and 2 (16-bit)."
        )

    count = int(np.prod(grid)) * out_ch
    values = np.frombuffer(data, dtype=dtype, count=count, offset=offset + 20)
    table = values.astype(np.float64).reshape(grid + (out_ch,)) / scale
    return CLUT(grid=grid, table=table)


# -- PCS decoding -----------------------------------------------------------


def _apply_curves(x: np.ndarray, curves: Sequence[ToneCurve], label: str) -> np.ndarray:
    if x.shape[-1] != len(curves):
        raise UnsupportedICCProfileError(
            f"{label} curves: {len(curves)} curves for {x.shape[-1]} channels."
        )
    return np.stack([curves[i](x[..., i]) for i in range(len(curves))], axis=-1)


def _decode_pcs(x: np.ndarray, encoding: str) -> np.ndarray:
    """Turn normalized PCS values in [0, 1] into PCS XYZ.

    The Lab encodings differ by tag type, and the difference is not cosmetic:
    ``lab_legacy16`` divides by 0xFF00 while ``lab_v4_16`` divides by 0xFFFF, a
    factor of 1.0039 on L*. At the highlight end of a tonal ramp that is roughly
    0.38 L*, which is enough on its own to change a FADGI tone response verdict.
    """
    if encoding == "xyz_u1f15":
        # u1Fixed15: 0x8000 encodes 1.0, so the range runs to just under 2.0.
        return x * (65535.0 / 32768.0)

    if encoding == "lab_legacy16":
        L = x[..., 0] * 65535.0 * 100.0 / 65280.0
        a = x[..., 1] * 65535.0 * 255.0 / 65280.0 - 128.0
        b = x[..., 2] * 65535.0 * 255.0 / 65280.0 - 128.0
    elif encoding in ("lab_v4_16", "lab_8"):
        L = x[..., 0] * 100.0
        a = x[..., 1] * 255.0 - 128.0
        b = x[..., 2] * 255.0 - 128.0
    else:
        raise UnsupportedICCProfileError(f"Unknown PCS encoding {encoding!r}")

    return _pcs_lab_to_xyz(np.stack([L, a, b], axis=-1))


def _pcs_lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """PCS Lab to PCS XYZ, against the ICC encoding of D50.

    PCS Lab is *defined* relative to the ICC D50 white point, so decoding must use
    it. Downstream the result is re-expressed against CIE D50 like every other
    value, which is what the reference measurement uses.
    """
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def inv(t: np.ndarray) -> np.ndarray:
        t3 = t**3
        return np.where(t3 > epsilon, t3, (116.0 * t - 16.0) / kappa)

    xr = inv(fx)
    yr = np.where(L > kappa * epsilon, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = inv(fz)
    return np.stack([xr, yr, zr], axis=-1) * np.asarray(ICC_PCS_D50)


# -- low-level tag reading --------------------------------------------------


def _sig(data: bytes, offset: int) -> str:
    return data[offset : offset + 4].decode("ascii", errors="replace")


def _read_tag_table(data: bytes) -> dict[str, tuple[int, int]]:
    count = struct.unpack_from(">I", data, 128)[0]
    if count > 1024:
        raise UnsupportedICCProfileError(f"Implausible ICC tag count {count}; file corrupt?")
    tags: dict[str, tuple[int, int]] = {}
    for i in range(count):
        base = 132 + i * 12
        if base + 12 > len(data):
            raise UnsupportedICCProfileError("ICC tag table runs past end of profile data.")
        sig, offset, size = struct.unpack_from(">4sII", data, base)
        name = sig.decode("ascii", errors="replace")
        if offset + size > len(data):
            raise UnsupportedICCProfileError(
                f"ICC tag {name!r} claims bytes {offset}..{offset + size} "
                f"but the profile is {len(data)} bytes."
            )
        tags[name] = (offset, size)
    return tags


def _read_xyz(data: bytes, loc: tuple[int, int]) -> np.ndarray:
    offset, size = loc
    if size < 20 or _sig(data, offset) != "XYZ ":
        raise UnsupportedICCProfileError(
            f"Expected an XYZType tag at offset {offset}, found {_sig(data, offset)!r}."
        )
    x, y, z = struct.unpack_from(">iii", data, offset + 8)
    return np.array([x / _S15F16, y / _S15F16, z / _S15F16], dtype=np.float64)


def _read_curve(data: bytes, loc: tuple[int, int], name: str) -> ToneCurve:
    offset, size = loc
    kind = _sig(data, offset)

    if kind == "curv":
        count = struct.unpack_from(">I", data, offset + 8)[0]
        if count == 0:
            return ToneCurve(kind="identity")
        if count == 1:
            raw = struct.unpack_from(">H", data, offset + 12)[0]
            return ToneCurve(kind="gamma", gamma=raw / _U8F8)
        need = 12 + count * 2
        if need > size:
            raise UnsupportedICCProfileError(
                f"{name}: curve declares {count} points but tag is only {size} bytes."
            )
        values = np.frombuffer(data, dtype=">u2", count=count, offset=offset + 12)
        return ToneCurve(kind="table", table=values.astype(np.float64) / 65535.0)

    if kind == "para":
        func = struct.unpack_from(">H", data, offset + 8)[0]
        counts = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
        if func not in counts:
            raise UnsupportedICCProfileError(
                f"{name}: parametricCurveType function {func} is not defined in ICC.1:2010."
            )
        n = counts[func]
        raw = struct.unpack_from(f">{n}i", data, offset + 12)
        return ToneCurve(kind="parametric", para_type=func, params=tuple(v / _S15F16 for v in raw))

    raise UnsupportedICCProfileError(
        f"{name}: unsupported tone curve type {kind!r} (expected 'curv' or 'para')."
    )


def _read_description(data: bytes, tags: dict[str, tuple[int, int]]) -> str | None:
    if "desc" not in tags:
        return None
    offset, size = tags["desc"]
    kind = _sig(data, offset)

    if kind == "desc":  # textDescriptionType (ICC v2)
        length = struct.unpack_from(">I", data, offset + 8)[0]
        if length == 0:
            return None
        text = data[offset + 12 : offset + 12 + length]
        return text.split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip() or None

    if kind == "mluc":  # multiLocalizedUnicodeType (ICC v4)
        count, _rec_size = struct.unpack_from(">II", data, offset + 8)
        if count == 0:
            return None
        _lang, _country, str_len, str_off = struct.unpack_from(">2s2sII", data, offset + 16)
        start = offset + str_off
        return (
            data[start : start + str_len]
            .decode("utf-16-be", errors="replace")
            .rstrip("\x00")
            .strip()
            or None
        )

    if kind == "text":
        return (
            data[offset + 8 : offset + size]
            .split(b"\x00", 1)[0]
            .decode("latin-1", errors="replace")
            .strip()
            or None
        )

    return None
