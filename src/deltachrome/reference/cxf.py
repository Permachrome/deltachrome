"""CxF3 (Color Exchange Format) reference reader.

CxF3 is ISO 17972-1 and is what several target vendors ship instead of CGATS --
LaserSoft's ISO 12641-2 targets, for one. It is XML rather than the CGATS tabular
form, but it carries the same thing: per-patch colorimetry plus the measurement
conditions that make it interpretable.

The output is the same :class:`MeasurementSet` the CGATS reader produces, with the
CxF measurement conditions mapped onto the CGATS keyword names the rest of the
engine already reads. That way format is purely an input concern: nothing
downstream needs to know which one a reference arrived in.

The same refusals apply. A repeated ``Name`` is an error, not a last-one-wins
merge, and a file with no colorimetry at all is rejected rather than silently
producing an empty comparison.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree

from deltachrome.errors import CGATSParseError, DuplicateSampleError
from deltachrome.reference.cgats import MeasurementSet, PatchMeasurement, _duplicate_report

CXF_NAMESPACES = (
    "http://colorexchangeformat.com/CxF3-core",
    "http://colorexchangeformat.com/v2",
)

#: CxF measurement conditions mapped onto the CGATS keywords the engine reads, so a
#: CxF reference and a CGATS reference are interchangeable downstream.
_OBSERVER_TEXT = {"2_degree": "2", "10_degree": "10"}


def read_cxf(path: str | Path) -> MeasurementSet:
    """Read a CxF3 file into a :class:`MeasurementSet`."""
    path = Path(path)
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise CGATSParseError(f"{path.name}: not well-formed XML -- {exc}") from exc
    result = parse_cxf_root(root)
    result.source_path = path
    return result


def parse_cxf(text: str) -> MeasurementSet:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise CGATSParseError(f"Not well-formed XML -- {exc}") from exc
    return parse_cxf_root(root)


def parse_cxf_root(root: ElementTree.Element) -> MeasurementSet:
    if _folded(root.tag) != "CxF":
        raise CGATSParseError(
            f"Root element is <{_local(root.tag)}>, expected <CxF>. This does not look "
            "like a CxF file."
        )

    metadata = _file_information(root)
    metadata.update(_color_specification(root))

    samples: dict[str, PatchMeasurement] = {}
    duplicates: dict[str, list[dict[str, float | str]]] = {}
    fields: list[str] = ["SAMPLE_NAME"]
    warnings: list[str] = []

    objects = [e for e in root.iter() if _folded(e.tag) == "Object"]
    if not objects:
        raise CGATSParseError(
            "CxF file contains no <Object> entries, so there is nothing to compare "
            "against."
        )

    for element in objects:
        name = (element.get("Name") or element.get("Id") or "").strip()
        if not name:
            continue
        lab = _triplet(element, "ColorCIELab", ("L", "A", "B"))
        xyz = _triplet(element, "ColorIEXYZ", ("X", "Y", "Z"))
        spectral = _spectrum(element)

        raw: dict[str, float | str] = {"SAMPLE_NAME": name}
        if lab:
            raw.update({"LAB_L": lab[0], "LAB_A": lab[1], "LAB_B": lab[2]})
        if xyz:
            raw.update({"XYZ_X": xyz[0], "XYZ_Y": xyz[1], "XYZ_Z": xyz[2]})

        measurement = PatchMeasurement(
            sample_name=name, lab=lab, xyz=xyz, spectral=spectral or None, raw=raw
        )
        if name in samples:
            duplicates.setdefault(name, [samples[name].raw]).append(raw)
            continue
        samples[name] = measurement

    if duplicates:
        raise DuplicateSampleError(_duplicate_report(duplicates))

    if not any(m.has_colorimetry for m in samples.values()):
        raise CGATSParseError(
            "CxF file contains objects but none carry Lab, XYZ or spectral data."
        )

    first = next(iter(samples.values()))
    if first.lab:
        fields += ["LAB_L", "LAB_A", "LAB_B"]
    if first.xyz:
        fields += ["XYZ_X", "XYZ_Y", "XYZ_Z"]
    if first.spectral:
        fields += [f"SPECTRAL_NM_{int(nm)}" for nm in sorted(first.spectral)]


    return MeasurementSet(
        samples=samples, metadata=metadata, fields=fields, warnings=warnings
    )


def _local(tag: str) -> str:
    """The element's own name, with any namespace stripped and nothing else changed.

    Kept truthful because it is what error messages report: telling somebody their
    root element is ``<Color...>`` when their file says ``<Colour...>`` sends them
    looking for text that is not in it.
    """
    return tag.rsplit("}", 1)[-1]


def _folded(tag: str) -> str:
    """The element's name with British spelling folded onto the schema's.

    ISO 17972-1 spells its elements ``ColorValues``, ``ColorCIELab``,
    ``ColorIEXYZ``; **LaserSoft ship ``ColourValues``, ``ColourCIELab`` and
    ``ColourIEXYZ``**, and their file is otherwise perfectly well formed. Read
    strictly, none of those elements exists, so a file carrying 864 patches of
    Lab *and* XYZ *and* spectra looked to this reader like a file carrying only
    spectra -- ``ReflectanceSpectrum`` being the one element spelled the same
    either way.

    Nothing went visibly wrong, which is the point: the spectra were integrated
    instead, and agreed with the Lab the file states to 0.066 Delta E2000 mean over
    all 864 patches. The cost was only that the file's own stated values were being
    quietly set aside -- and that a British-spelled file carrying *no* spectra would
    have been refused outright as having no colorimetry at all.

    The fold is done on comparison rather than by listing both spellings at each
    site, so a spelling missed in one helper cannot behave differently from the same
    spelling in another.
    """
    return _local(tag).replace("Colour", "Color")


def _find(parent: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return next((e for e in parent.iter() if _folded(e.tag) == name), None)


def _text(parent: ElementTree.Element, name: str) -> str | None:
    element = _find(parent, name)
    return element.text.strip() if element is not None and element.text else None


def _triplet(
    element: ElementTree.Element, container: str, keys: tuple[str, str, str]
) -> tuple[float, float, float] | None:
    node = _find(element, container)
    if node is None:
        return None
    values = []
    for key in keys:
        child = next((c for c in node if _folded(c.tag).upper() == key.upper()), None)
        if child is None or not child.text:
            return None
        try:
            values.append(float(child.text))
        except ValueError:
            return None
    return (values[0], values[1], values[2])


def _spectrum(element: ElementTree.Element) -> dict[float, float]:
    node = next(
        (e for e in element.iter() if _folded(e.tag).endswith("Spectrum")),
        None,
    )
    if node is None or not node.text:
        return {}
    try:
        start = float(node.get("StartWL", 380))
        increment = float(node.get("Increment", 10))
    except ValueError:
        return {}
    try:
        values = [float(v) for v in node.text.split()]
    except ValueError:
        return {}
    return {start + i * increment: v for i, v in enumerate(values)}


def _file_information(root: ElementTree.Element) -> dict[str, str]:
    info = _find(root, "FileInformation")
    metadata: dict[str, str] = {}
    if info is None:
        return metadata

    for cxf_name, keyword in (
        ("Creator", "ORIGINATOR"),
        ("Description", "DESCRIPTOR"),
        ("CreationDate", "CREATED"),
        ("Manufacturer", "MANUFACTURER"),
    ):
        value = _text(info, cxf_name)
        if value:
            metadata[keyword] = value

    # Vendor <Tag Name=.. Value=../> entries carry the target serial and type, which
    # is exactly what a barcode on the chart would key against.
    for tag in info.iter():
        if _folded(tag.tag) != "Tag":
            continue
        name, value = tag.get("Name"), tag.get("Value")
        if not name or value is None:
            continue
        key = re.sub(r"^\d+_", "", name).upper()
        metadata[f"CXF_{key}"] = value
        if key in ("SERIAL", "SERIALNUMBER"):
            metadata["TARGET_SERIAL"] = value
        elif key == "TARGETTYPE":
            metadata["TARGET_TYPE"] = value
        elif key == "MATERIAL":
            metadata["MATERIAL"] = value
        elif key == "VALIDATION":
            metadata["VALIDATION"] = value
    return metadata


def _color_specification(root: ElementTree.Element) -> dict[str, str]:
    spec = _find(root, "ColorSpecification")
    metadata: dict[str, str] = {}
    if spec is None:
        return metadata

    illuminant = _text(spec, "Illuminant")
    observer = _text(spec, "Observer")
    method = _text(spec, "Method")
    measurement_type = _text(spec, "MeasurementType")

    if illuminant:
        metadata["ILLUMINATION_NAME"] = illuminant
    if observer:
        metadata["OBSERVER_ANGLE"] = _OBSERVER_TEXT.get(observer.lower(), observer)
    if measurement_type:
        metadata["MEASUREMENT_TYPE"] = measurement_type

    # MEASUREMENT_SOURCE is where MeasurementSet.measurement_mode looks for M0/M1/M2.
    source = []
    if method:
        source.append(f"MeasurementCondition={method}")
    if illuminant:
        source.append(f"Filter={illuminant}")
    if source:
        metadata["MEASUREMENT_SOURCE"] = " ".join(source)

    device = _find(spec, "Device")
    if device is not None:
        model = _text(device, "Model") or _text(device, "DeviceModel")
        maker = _text(device, "Manufacturer")
        instrument = " ".join(x for x in (maker, model) if x)
        if instrument:
            metadata["INSTRUMENTATION"] = instrument
    return metadata
