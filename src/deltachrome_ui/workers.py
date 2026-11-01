"""Work that must not run on the UI thread.

Decoding a master is not a quick operation. Measured on this repository's own
samples: the 577 MB ISA 4x5 TIFF maps in almost instantly, but the 46 MB
ColorChecker SG JPEG 2000 takes **24.6 seconds** to decode, because JP2 is
compressed and TIFF is not. JP2 is a preservation master format across the
sector, so that is a normal file, not a pathological one. Doing it inline would
freeze the window for half a minute on a routine open.
"""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QThread, Signal

from deltachrome.errors import DeltachromeError, MissingICCProfileError
from deltachrome_ui.imageproxy import ImageProxy


class LoadFailure(Exception):
    """A load that failed for a reason worth explaining rather than logging."""

    def __init__(self, headline: str, detail: str) -> None:
        super().__init__(headline)
        self.headline = headline
        self.detail = detail


def describe_failure(exc: Exception, path: Path) -> LoadFailure:
    """Turn an engine exception into something worth putting in front of a person.

    :class:`~deltachrome.errors.MissingICCProfileError` gets its own text because
    it is the engine working correctly, not breaking. The refusal is deliberate --
    assuming sRGB for an Adobe RGB or ECI RGB master yields Delta E values that are
    wrong by many units while looking entirely reasonable -- and a dialog that
    renders that as an error with a traceback invites the user to go looking for a
    setting that will make it go away. There isn't one, and there shouldn't be.
    """
    if isinstance(exc, MissingICCProfileError):
        return LoadFailure(
            "This file has no embedded ICC profile.",
            f"{path.name} carries no color profile, so there is no way to know what "
            "its numbers mean. Deltachrome refuses rather than assuming sRGB: masters "
            "in this field are routinely Adobe RGB, ProPhoto or ECI RGB, and guessing "
            "produces Delta E figures that look entirely reasonable and are wrong by "
            "many units.\n\nEmbed the correct profile at source and open it again.",
        )
    if isinstance(exc, DeltachromeError):
        return LoadFailure(f"{path.name} could not be opened.", str(exc))
    return LoadFailure(
        f"{path.name} could not be opened.",
        f"{type(exc).__name__}: {exc}",
    )


class ImageLoader(QObject):
    """Loads one image and builds its display proxy, off the UI thread."""

    progress = Signal(str)
    #: The proxy and the path it came from. The path travels *with* the signal so
    #: the handler can be a plain bound method: a connection to a lambda has no
    #: QObject receiver, so PySide cannot tell which thread to deliver on and runs
    #: it directly -- on the worker's thread. That is measured, not surmised, and
    #: it meant the shell's load handler was calling widget methods off the GUI
    #: thread. See ``test_no_worker_signal_is_connected_to_a_lambda``.
    loaded = Signal(object, object)  # ImageProxy, Path
    failed = Signal(object)  # LoadFailure

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)

    def run(self) -> None:
        from deltachrome.imaging.loader import load_image

        try:
            self.progress.emit("Loading File\u2026")
            # Opened without a profile where it has none: such a capture can be
            # placed and a camera profile built from it, and the placement
            # screen withholds every measurement that needs the profile.
            image = load_image(self.path, profile_required=False)
            self.progress.emit("Loading File\u2026")
            proxy = ImageProxy(image)
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            self.failed.emit(describe_failure(exc, self.path))
            return
        self.loaded.emit(proxy, self.path)


class ThumbnailLoader(QObject):
    """A small picture of a capture, off the UI thread.

    For the frame map on a capture reopened from the library, which has the
    outlines from the record and no pixels. Worst measured case is 0.61 s on a
    JPEG 2000 master -- short enough that no button, busy strip or
    second-press guard is wanted, and long enough that it must not run inline.

    The fingerprint travels *with* the result so the handler can drop a
    thumbnail whose capture is no longer on screen. There is no ``failed``
    signal: a thumbnail that cannot be read is not a failure of the screen, and
    the map goes on drawing its outlines exactly as it does before one arrives.

    **``finished`` is emitted whatever happens**, and that is not bookkeeping.
    :func:`run_in_thread` ends the thread on ``loaded``, ``finished`` or
    ``failed``, so a worker that stays silent on the path it takes most often --
    a master that has moved -- leaks its thread and its own record, and the
    owner holds both for the life of the window. (Verified by pumping the
    owner's loop rather than by ``QThread.wait``: ``quit`` is a queued delivery
    to the thread object, which lives on the *owner's* thread, so waiting on it
    there blocks the very loop that has to deliver it.)
    """

    #: The picture, and the capture key it belongs to. Only on success.
    loaded = Signal(object, object)  # QImage, str
    #: Emitted in every outcome, including the ones that produce no picture.
    finished = Signal()

    def __init__(self, path: Path, key: str) -> None:
        super().__init__()
        self.path = Path(path)
        self.key = key

    def run(self) -> None:
        from deltachrome_ui.imageproxy import read_thumbnail

        try:
            image = read_thumbnail(self.path)
            if image is not None:
                self.loaded.emit(image, self.key)
        finally:
            self.finished.emit()


class TargetIdentifier(QObject):
    """Works out which chart the capture holds, off the UI thread.

    Several seconds on a large master -- it runs a finder per placement method and
    then reads the patches of every candidate that could be placed -- so it cannot
    run inline. It hands back an ``Identification`` in every outcome, including the
    two that are not an answer: no target fitting, and more than one fitting equally.

    **One per chart, not one per capture.** A frame may hold several, and they need
    not be the same model: the search is per region and each region gets its own
    answer, so a device-level target beside a colour chart is two entries rather than
    a contest between them. A frame holding one chart yields a list of one, which is
    what every capture but two in this repository does.

    **It also reads the chart's printed code**, when asked, and that is the same
    question one level down: identification says *which model* is in the frame, a
    code says *which physical unit*, and reference data is per unit. It was read at
    run time instead, which is the worst place for it -- a capture whose code
    matches nothing in the bank raised ``MissingReferenceError`` after the decode,
    which is a dialog arrived at after a wait and dismissable only. Read here, the
    pairing is on the setup screen beside the chart it is about, before anything is
    measured.

    The cost is small and is why it can sit in this pass at all. A whole-image pass
    at about a 2600 px long edge is **0.1 s** and reads the ColorChecker SG's QR
    outright; only a capture where that finds nothing pays for tiling the master,
    which is 1.7 s over 49 tiles on the LaserSoft 4x5. Against an 8.6 s decode
    neither is worth a screen of its own. It is skipped entirely when the bank is
    empty, there being nothing a code could pair with.
    """

    progress = Signal(str)
    finished = Signal(list)  # list[Identification], never empty
    #: The capture's printed codes, or ``None`` where no scan was asked for.
    #: Separate from ``finished`` because it is a different fact about the capture
    #: and a caller may want one without the other.
    scanned = Signal(object)  # CodeScanResult | None
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, image, scan_codes: bool = False, region=None,
                 target_ids=None) -> None:
        super().__init__()
        self.image = image
        self.scan_codes = bool(scan_codes)
        #: The targets to choose among, or ``None`` for every installed one --
        #: Settings' list of owned targets, read by the caller as the search starts.
        self.target_ids = None if target_ids is None else list(target_ids)
        #: ``(x0, y0, x1, y1)`` the operator drew round one chart, or None to look
        #: at the whole frame. Identification already takes the region to look in --
        #: that is how a frame holding several charts gets an answer per region --
        #: so a drawn rectangle needs nothing here but somewhere to put the box.
        self.region = region

    def run(self) -> None:
        from deltachrome.imaging.detect import (
            Identification, extent_from_corners, identify_target, identify_targets,
        )

        scan = self._scan()
        self.scanned.emit(scan)
        try:
            self.progress.emit("Detecting Target\u2026")
            if self.region is not None:
                x0, y0, x1, y1 = self.region
                extent = extent_from_corners(
                    self.image, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                )
                self.finished.emit(self._noted([identify_target(
                    self.image, self.target_ids, extent=extent
                )]))
                return
            # Where a code was decoded, the search is told where. A printed code
            # block is exactly what the chart search looks for -- a compact region
            # of high-contrast structure unlike its surround -- so the
            # ColorChecker SG's own QR is otherwise offered as a second target in
            # its own frame, and nothing about the pixels could say otherwise.
            found = identify_targets(
                self.image, self.target_ids,
                code_boxes=(scan.boxes if scan is not None else ()),
            )
            self.finished.emit(self._noted(found or [Identification(reason=(
                "No installed target could be found in this capture. Choose the "
                "chart yourself, and place it by hand if it cannot be found."
            ))]))
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            # Choosing the chart by hand is still available and is what the operator
            # did before this existed, so a failure here is a refusal like any other.
            self.finished.emit([Identification(reason=(
                f"Working out which chart this is stopped: {type(exc).__name__}: "
                f"{exc}. Choose it yourself."
            ))])

    def _noted(self, results: list) -> list:
        """Say that targets were left out, wherever the search did not name one.

        A target turned off in Settings is the likeliest reason a chart that is
        plainly there was not found -- one borrowed, or bought since -- and a list
        of owned targets that cannot say so becomes a quiet reason charts go
        missing. Only where nothing was named: a named chart is checked on the
        canvas, and a note on every run would be read past.
        """
        from deltachrome.imaging.detect import identifiable_targets

        if self.target_ids is None:
            return results
        hidden = len(set(identifiable_targets()) - set(self.target_ids))
        if not hidden:
            return results
        note = (
            "1 target is turned off in Settings and was not searched for."
            if hidden == 1 else
            f"{hidden} targets are turned off in Settings and were not searched for."
        )
        for result in results:
            if not result.found:
                result.notes.append(note)
        return results

    def _scan(self):
        """The capture's codes, or ``None``.

        A failure here is never a failure of the run: a code is a convenience for
        pairing, and the two other reference modes do not need one. So this
        reports nothing rather than raising, exactly as the identification below
        refuses rather than taking the open down with it.
        """
        if not self.scan_codes:
            return None
        from deltachrome.imaging.codes import scan_codes  # noqa: PLC0415

        try:
            self.progress.emit("Reading the Target\u2019s Code\u2026")
            return scan_codes(self.image, tiled=True)
        except Exception:  # noqa: BLE001 -- a scan that fails pairs nothing
            return None


class PlacementDetector(QObject):
    """Finds the chart in an already-decoded capture, off the UI thread.

    Off the thread for the same reason the decode is: the search runs a threshold
    sweep over four corner windows of a hundred-megapixel master, which is around
    a second and occasionally several. A window that stops repainting between
    choosing a target and seeing the chart placed would read as a hang at exactly
    the moment the operator is waiting to be shown something.

    **Given a region, it searches there instead of hunting for one.** That is the
    same mechanism a batch uses to carry one checked placement across a hundred
    frames, pointed at a rectangle somebody dragged rather than at a previous
    run: ``chart_extent`` returns the whole frame whenever a chart is a small part
    of a busy one, and every finder's size gate is a fraction of the region, so
    the whole-frame answer puts the smallest admissible patch *above* the largest
    the chart prints. A drawn rectangle fixes that at the cause.
    """

    progress = Signal(str)
    #: Every placement the frame supports, in reading order, and never empty: a
    #: refusal is one entry saying why. A list rather than a Detection because a
    #: frame can hold several charts -- the QA-62 is shot four to a board -- and
    #: the screen has to be able to offer them.
    finished = Signal(list)  # list[Detection]
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, image, target, region=None) -> None:
        super().__init__()
        self.image = image
        self.target = target
        #: ``(x0, y0, x1, y1)`` the operator drew round one chart, or None to
        #: search the whole frame. A region means one chart is wanted -- the
        #: operator has said which -- so it takes the single-placement path.
        self.region = region

    def run(self) -> None:
        from deltachrome.imaging.detect import (
            detect_placement_near, detect_placements,
        )

        try:
            self.progress.emit("Detecting Target\u2026")
            if self.region is None:
                found = detect_placements(self.image, self.target)
            else:
                x0, y0, x1, y1 = self.region
                found = [detect_placement_near(
                    self.image, self.target,
                    [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                )]
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            # A detector that raises must not take the run down with it: placing
            # the chart by hand is still available and is what the operator did
            # before this existed, so a failure here is a refusal like any other.
            from deltachrome.imaging.detect import Detection

            self.finished.emit([Detection(
                method=self.target.fiducials.method,
                reason=(
                    f"Automatic placement stopped: {type(exc).__name__}: {exc}. "
                    "Place the chart by hand."
                ),
            )])
            return
        self.finished.emit(found)


class SampleRegionFinder(QObject):
    """Works out every region a run would read, off the UI thread.

    Off the thread because the picture is deliberately the real thing. The edge
    ROIs drawn are the ones the measurement cuts, refinement and all, on the
    argument that a picture of a region the run does not use would confirm a
    placement that was never going to be measured -- and that refinement is the
    expensive half. Measured on the committed placements: **898 ms** on the ISA
    transmissive 4x5, of which 818 is ``extract_edges`` snapping ten quarter
    discs; 326 ms on the ColorChecker SG's 140 patches; 223 on the Golden Thread.

    Inline, that landed on the one gesture this screen exists for. The overlay is
    rebuilt 250 ms after the corners stop moving, so placing a corner *carefully*
    -- nudge, look, nudge -- put a near-second freeze between every nudge and the
    next, with the mouse events queueing behind it and the handle then jumping to
    catch up. That is the stutter, and it got worse the more precisely somebody
    tried to work.

    The token travels with the result so a stale answer can be dropped: a drag
    outruns this by a wide margin and the corners it was computed for are the only
    thing that makes it worth drawing.
    """

    finished = Signal(object)  # SampleRegions
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, image, target, corners, token) -> None:
        super().__init__()
        self.image = image
        self.target = target
        self.corners = corners
        self.token = token

    def run(self) -> None:
        from deltachrome_ui.placement_view import sample_regions

        try:
            polygons, flagged, edges = sample_regions(
                self.image, self.target, self.corners
            )
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            # A placement this cannot sample is not a failed run: the corners are
            # still there to be moved, and a crossed quad is the ordinary way to
            # arrive here. The canvas is told there is nothing to draw and the
            # panel says why.
            self.finished.emit(SampleRegions(
                token=self.token, polygons=[], flagged=set(), edges=[],
                reason=f"{type(exc).__name__}: {exc}",
            ))
            return
        self.finished.emit(SampleRegions(
            token=self.token, polygons=polygons, flagged=flagged, edges=edges,
        ))


@dataclass
class SampleRegions:
    """Every region one placement would read, and the placement it belongs to."""

    #: Identifies the capture, target and corners this was computed for. The view
    #: draws it only while that is still what is on screen.
    token: object
    polygons: list
    flagged: set
    edges: list
    #: Why nothing could be sampled, when nothing could be.
    reason: str = ""


class AnalysisRunner(QObject):
    """Resolves the reference and runs the analysis, off the UI thread.

    Both, in one worker, because they are one wait from the operator's point of
    view and because the first feeds the second.

    The already-decoded image goes to both steps. That is the whole reason
    ``resolve_reference`` and ``analyze`` accept a ``LoadedImage``: the master is
    decoded once for the canvas and never again. ``scan`` is the same argument
    made once more: the codes were read on the way to identifying the chart, so
    the pairing shown on the setup screen and the pairing the run uses are the
    same reading rather than two scans that could in principle disagree.
    """

    progress = Signal(str)
    finished = Signal(object)  # AnalysisResult
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, image, target_id: str, profile_id: str, corners,
                 reference_mode: str, reference_choice_path=None, options=None,
                 placement_notes=None, scan=None, placed: str = "by_hand",
                 identified: bool = False, searched_targets=None) -> None:
        super().__init__()
        self.image = image
        #: How the corners came to be, whether the chart was named by the search,
        #: and what it was named from. Written into the record's geometry.
        self.placed = placed
        self.identified = identified
        self.searched_targets = searched_targets
        #: What the search had to say about the corners it produced -- a
        #: refinement that could not run, features that disagreed. That the
        #: placement was automatic at all is ``placed``, in the record's geometry.
        self.placement_notes = list(placement_notes or [])
        self.target_id = target_id
        self.profile_id = profile_id
        self.corners = list(corners)
        self.reference_mode = reference_mode
        self.reference_choice_path = reference_choice_path
        self.options = dict(options or {})
        #: The capture's printed codes, already read. See the class docstring.
        self.scan = scan

    def run(self) -> None:
        from deltachrome.pipeline import analyze
        from deltachrome.reference.resolve import resolve_reference
        from deltachrome.targets.registry import load_target

        try:
            target = load_target(self.target_id)
            self.progress.emit(
                "Pairing the reference…" if self.scan is not None
                else "Checking the chart's printed code…"
            )
            choice = resolve_reference(
                target=target,
                image=self.image,
                explicit_path=(
                    self.reference_choice_path if self.reference_mode == "file" else None
                ),
                library_dir=(
                    self.reference_choice_path if self.reference_mode == "library" else None
                ),
                scan=self.scan,
            )
            self.progress.emit("Sampling patches and measuring…")
            result = analyze(
                self.image,
                target=target,
                reference_path=choice.path,
                profile=self.profile_id,
                corners=self.corners,
                extra_warnings=list(choice.warnings) + self.placement_notes,
                placed=self.placed,
                identified=self.identified,
                searched_targets=self.searched_targets,
                **self.options,
            )
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            self.failed.emit(describe_analysis_failure(exc))
            return
        self.finished.emit(result)


@dataclass
class ProfileSetup:
    """What the profile screen opens on: every patch, sampled once, and its file.

    ``data`` holds every patch with a reference value and nothing left out but
    what the sampler flagged; the screen applies the operator's own choices to
    it (:func:`~deltachrome.profiling.select_patches`) without sampling again.
    """

    data: object  # ProfilingSet
    reference_name: str
    #: The reference's own file name and a short form of the capture's identity,
    #: written into the profile's embedded .ti3.
    keywords: dict
    warnings: list
    encoding: object  # Encoding | None
    #: How many readings merging repeats would leave, or None for no repeats.
    merged_count: int | None


class ProfileGatherer(QObject):
    """Pairs the reference and samples every patch, off the UI thread.

    The same reference resolution a run uses, so the file this screen names is
    the file the QC run on the same capture would have used -- and then the
    profile's own stricter rule about which references a profile may be built
    against.
    """

    progress = Signal(str)
    finished = Signal(object)  # ProfileSetup
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, image, target_id: str, corners, reference_mode: str,
                 reference_choice_path=None, scan=None, image_path=None) -> None:
        super().__init__()
        self.image = image
        self.target_id = target_id
        self.corners = list(corners)
        self.reference_mode = reference_mode
        self.reference_choice_path = reference_choice_path
        self.scan = scan
        self.image_path = image_path

    def run(self) -> None:
        from deltachrome.profiling import (
            check_reference, gather_patches, merge_repeats, repeat_families,
        )
        from deltachrome.profiling.encoding import detect_encoding
        from deltachrome.reference import read_reference
        from deltachrome.reference.resolve import resolve_reference
        from deltachrome.results.models import file_fingerprint
        from deltachrome.targets.registry import load_target

        try:
            target = load_target(self.target_id)
            self.progress.emit("Pairing the reference…")
            choice = resolve_reference(
                target=target,
                image=self.image,
                explicit_path=(
                    self.reference_choice_path if self.reference_mode == "file" else None
                ),
                library_dir=(
                    self.reference_choice_path if self.reference_mode == "library" else None
                ),
                scan=self.scan,
            )
            if choice.path is None:
                raise DeltachromeError(
                    f"No reference file for {target.display_name}. A profile is "
                    "built against the chart's own measurement."
                )
            reference = read_reference(choice.path)
            name = Path(choice.path).name
            warnings = list(choice.warnings) + list(reference.warnings)
            warnings += check_reference(reference, name)
            self.progress.emit("Sampling patches…")
            data = gather_patches(self.image, target, reference, self.corners)
            warnings += data.warnings
            merged = merge_repeats(data)
            fingerprint = (file_fingerprint(self.image_path) or {}) if self.image_path else {}
            setup = ProfileSetup(
                data=data,
                reference_name=name,
                keywords={
                    "target": target.target_id,
                    "reference": name,
                    "capture": Path(self.image_path).name if self.image_path else "",
                    "capture_digest": fingerprint.get("digest", ""),
                },
                warnings=warnings,
                encoding=detect_encoding(data),
                merged_count=(len(merged.used) if repeat_families(data.used) else None),
            )
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            self.failed.emit(_profiling_failure(exc))
            return
        self.finished.emit(setup)


class ProfileBuilder(QObject):
    """Runs colprof -- once, or once per candidate under Automatic -- off the UI thread.

    Every build is a separate process, so the wait is long and has a shape:
    under Automatic the progress names the candidate being tried and how many
    there are, which on the LaserSoft is eleven over three minutes.
    """

    progress = Signal(str)
    finished = Signal(object)  # BuiltProfile
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, data, options, colprof, keywords, scale_for,
                 types=None, qualities=None, automatic: bool = False) -> None:
        super().__init__()
        self.data = data
        self.options = options
        self.colprof = colprof
        self.keywords = dict(keywords)
        self.scale_for = scale_for
        self.types = types
        self.qualities = qualities
        self.automatic = automatic

    def _tried(self, kind: str, quality: str, number: int, count: int) -> None:
        from deltachrome.profiling.colprof import profile_type_label

        self.progress.emit(
            f"Trying {profile_type_label(kind)}, {quality.title()} · {number} of {count}"
        )

    def run(self) -> None:
        from deltachrome.profiling import build_automatic, build_profile

        try:
            if self.automatic:
                built = build_automatic(
                    self.data, self.options, colprof=self.colprof,
                    keywords=self.keywords, scale_for=self.scale_for,
                    types=self.types, qualities=self.qualities, progress=self._tried,
                )
            else:
                self.progress.emit("Building the profile…")
                built = build_profile(
                    self.data, self.options, colprof=self.colprof,
                    keywords=self.keywords, scale_for=self.scale_for,
                )
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            self.failed.emit(_profiling_failure(exc))
            return
        self.finished.emit(built)


def _profiling_failure(exc: Exception) -> LoadFailure:
    if isinstance(exc, DeltachromeError):
        return LoadFailure("The profile could not be built.", str(exc))
    return LoadFailure("The profile could not be built.", f"{type(exc).__name__}: {exc}")


class ChartsRunner(QObject):
    """Every chart in one frame, analyzed off the UI thread and combined.

    **Not one run holding several charts.** A run is one capture, one target, one
    placement, one verdict, and every record in the library is that shape; this is
    that run repeated, and the results are combined afterwards by the same code a
    session assembled from the library uses. What is shared is the decode, which is
    the expensive part -- ten seconds for a 101-megapixel master, paid once rather
    than five times.

    Each chart carries its own target and its own reference, because a frame may hold
    a device-level target beside a colour chart and a per-unit file describes one
    physical chart. Reference resolution therefore runs per chart, not once.

    A chart that fails is reported and the rest still run: four good charts and a
    named failure is a better answer than nothing, and a board is shot precisely
    because one chart is not the whole story.
    """

    progress = Signal(str)
    finished = Signal(object)  # (list[AnalysisResult], list[str])
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, image, charts, profile_id: str, reference_mode: str,
                 reference_choice_path=None, options=None, notes=None,
                 scan=None, identified: bool = False,
                 searched_targets=None) -> None:
        super().__init__()
        self.image = image
        #: ``(target, corners, notes, placed)`` per chart, in reading order.
        self.charts = list(charts)
        self.identified = identified
        self.searched_targets = searched_targets
        self.profile_id = profile_id
        self.reference_mode = reference_mode
        self.reference_choice_path = reference_choice_path
        self.options = dict(options or {})
        self.notes = list(notes or [])
        self.scan = scan

    def run(self) -> None:
        from deltachrome.pipeline import analyze
        from deltachrome.reference.resolve import resolve_reference

        results, failures = [], []
        total = len(self.charts)
        for number, (target, corners, chart_notes, placed) in enumerate(
            self.charts, start=1
        ):
            self.progress.emit(
                f"Chart {number} of {total} — {target.display_name}…"
            )
            try:
                choice = resolve_reference(
                    target=target,
                    image=self.image,
                    explicit_path=(
                        self.reference_choice_path
                        if self.reference_mode == "file" else None
                    ),
                    library_dir=(
                        self.reference_choice_path
                        if self.reference_mode == "library" else None
                    ),
                    scan=self.scan,
                )
                results.append(analyze(
                    self.image,
                    target=target,
                    reference_path=choice.path,
                    profile=self.profile_id,
                    corners=list(corners),
                    extra_warnings=(
                        list(choice.warnings) + self.notes + list(chart_notes)
                    ),
                    placed=placed,
                    identified=self.identified,
                    searched_targets=self.searched_targets,
                    **self.options,
                ))
            except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
                failures.append(
                    f"Chart {number} ({target.display_name}): "
                    f"{describe_analysis_failure(exc).detail}"
                )
        if not results:
            self.failed.emit(describe_analysis_failure(
                RuntimeError("; ".join(failures) or "no chart could be analyzed")
            ))
            return
        self.finished.emit((results, failures))


class RecordEstablisher(QObject):
    """Reads a saved run's capture again, off the UI thread.

    A run reopened from the library holds every number and no pixels, so it can
    be read and combined into a session but not judged against another profile.
    The record already carries everything ``analyze`` needs to read the same
    capture the same way a second time -- the target, the placement, the
    reference and the run options it was judged under -- so this asks the
    operator nothing. What it produces is a fresh measurement in memory, after
    which the profile picker this report already has stops being a label.

    **It takes every chart the record holds, not the one on screen**, for the
    same reason :class:`Rescorer` does: a capture record carries one combined
    verdict, and a board re-established one card at a time would leave the
    others still numbers-only while the stepper claimed they were all live.

    A chart whose record cannot say enough of itself -- no corners, no target,
    no profile id, which only a record from long before this existed could be
    missing -- is skipped rather than failing the rest; the failure is reported
    the same way a chart that will not decode is.
    """

    progress = Signal(str)
    finished = Signal(object)  # (list[AnalysisResult], list[str])
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, image, charts: list[dict]) -> None:
        super().__init__()
        self.image = image
        #: Raw chart records, exactly as read off disk -- a bare run record
        #: wrapped in a list of one, or a capture record's own ``charts``.
        self.charts = list(charts)

    def run(self) -> None:
        from deltachrome.pipeline import analyze

        from deltachrome_ui.library import establish_kwargs

        results, failures = [], []
        total = len(self.charts)
        for number, record in enumerate(self.charts, start=1):
            name = (record.get("target") or {}).get("display_name") or f"chart {number}"
            self.progress.emit(f"Reading chart {number} of {total} — {name}…")
            kwargs = establish_kwargs(record)
            if kwargs is None:
                failures.append(
                    f"Chart {number} ({name}): this record does not carry enough "
                    "of its own placement to be read again."
                )
                continue
            try:
                results.append(analyze(self.image, **kwargs))
            except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
                failures.append(
                    f"Chart {number} ({name}): {describe_analysis_failure(exc).detail}"
                )
        if not results:
            self.failed.emit(describe_analysis_failure(
                RuntimeError("; ".join(failures) or "no chart could be read again")
            ))
            return
        self.finished.emit((results, failures))


class Rescorer(QObject):
    """Judges a reading already in memory against another profile, off the UI thread.

    Fast, and not fast enough to run inline. Measured: 0.21 s on the UTT's JPEG
    2000 capture and 0.48 s on the Golden Thread Device Level Target, against
    16.4 s and 1.26 s to analyze the same two files from scratch. A fifth of a
    second on the GUI thread is a window that visibly stops, and the number is a
    property of the chart -- twenty slanted edges in three colour records is the
    expensive case here, and nothing says a chart cannot carry more.

    It is a *worker* rather than a call for one further reason: it runs every
    metric plugin again, which is the same code the full analysis runs, and code
    that has always run off this thread should not start running on it because it
    happens to be reached by a shorter route.

    **It takes every chart of the capture, not the one on screen.** A capture
    record holds one combined verdict against one profile, and ``capture_record``
    requires it, so a board with card 3 judged at four star and the rest at three
    is not a thing that can be saved or combined. The control that chooses a
    profile sits on a report of one chart and is asking the question of the
    capture. Five charts is five rescores -- a second or two against the
    sixteen a re-read of the master would cost.
    """

    finished = Signal(object)  # list[AnalysisResult]
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, results, profile_id: str) -> None:
        super().__init__()
        self.results = list(results)
        self.profile_id = profile_id

    def run(self) -> None:
        from deltachrome.pipeline import rescore

        try:
            self.finished.emit(
                [rescore(result, self.profile_id) for result in self.results]
            )
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            self.failed.emit(describe_analysis_failure(exc))


class MeasurementFetcher(QObject):
    """Fetches an OpenQualia label's measurement and files it in the bank.

    Off the UI thread because it is the one thing this application does that
    waits on somebody else's server. The file is small -- 12 to 60 kB -- so the
    wait is usually a fraction of a second and occasionally a timeout, which is
    exactly the shape that must not block the window.

    It is started by a control the operator pressed. A capture is not consent to
    make an outbound request, so nothing on the identification path calls this.
    """

    #: The entry now in the bank, the label it was fetched for, and any caveat
    #: about the file itself -- a serial that disagrees with the unit requested.
    loaded = Signal(object, object, object)  # ReferenceEntry, OpenQualiaLabel, str|None
    failed = Signal(object)  # LoadFailure

    def __init__(self, label, directory: Path) -> None:
        super().__init__()
        self.label = label
        self.directory = Path(directory)

    def run(self) -> None:
        from deltachrome.reference.library import ReferenceLibrary
        from deltachrome.reference.openqualia import fetch_measurement

        try:
            fetched = fetch_measurement(self.label)
            self.directory.mkdir(parents=True, exist_ok=True)
            entry = ReferenceLibrary(self.directory).add_measurement(
                fetched.content,
                fetched.filename,
                # The unit that was *asked for*, never the serial the file states
                # about itself: there is a published measurement whose SERIAL
                # names a different chart than the one it was served for.
                bind=self.label.target_id,
                origin=self.label.url,
                note=fetched.serial_disagreement,
            )
        except DeltachromeError as exc:
            self.failed.emit(
                LoadFailure("The measurement could not be fetched.", str(exc))
            )
            return
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            self.failed.emit(
                LoadFailure(
                    "The measurement could not be fetched.",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return
        self.loaded.emit(entry, self.label, fetched.serial_disagreement)


def describe_analysis_failure(exc: Exception) -> LoadFailure:
    """Engine refusals are designs, not defects, and should not read as crashes."""
    from deltachrome.errors import (
        CapabilityError,
        DeltachromeError,
        GeometryError,
        MissingReferenceError,
    )

    if isinstance(exc, MissingReferenceError):
        return LoadFailure("There is nothing to measure against.", str(exc))
    if isinstance(exc, GeometryError):
        return LoadFailure("The corner placement could not be resolved.", str(exc))
    if isinstance(exc, CapabilityError):
        return LoadFailure("These guidelines ask for more than this target can do.",
                           str(exc))
    if isinstance(exc, DeltachromeError):
        return LoadFailure("The analysis stopped.", str(exc))
    return LoadFailure("The analysis stopped.", f"{type(exc).__name__}: {exc}")


def run_in_thread(worker: QObject, owner: QObject) -> QThread:
    """Move ``worker`` onto a new thread and start it.

    ``owner`` keeps both alive. The worker is the one that bites: ``moveToThread``
    requires it to have no parent, so nothing but the caller's local variable
    refers to it, and the moment that call returns Python is free to collect a
    worker that is still running. It survives a fast job and vanishes during a
    slow one, which is the worst way for a bug to behave.

    **Python owns the lifetime, not Qt.** An earlier version also connected
    ``deleteLater`` on both objects, which meant Qt destroyed the C++ objects
    while these records still held wrappers to them -- and touching one of those
    afterwards is a use-after-free, not an exception. Dropping the record instead
    releases the last reference and lets ordinary collection do it safely.

    The record is keyed by a token because it used to be found by value: the
    cleanup searched a list for its own ``(thread, worker)`` tuple, which compares
    PySide wrappers, and once a wrapper went stale the comparison quietly failed
    to match. The record was then never removed -- leaking the thread, the worker,
    and through the worker the whole decoded master, which for these files is
    hundreds of megabytes for every run of the session.
    """
    thread = QThread(owner)
    worker.setParent(None)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for signal in ("loaded", "finished", "failed"):
        if hasattr(worker, signal):
            getattr(worker, signal).connect(thread.quit)

    if not hasattr(owner, "_threads"):
        owner._threads = {}
    key = id(worker)
    owner._threads[key] = _Running(thread=thread, worker=worker)
    # Queued so the record -- and with it the last reference to the thread -- is
    # dropped on the owner's thread rather than inside the dying thread's own
    # finished handler.
    thread.finished.connect(
        lambda: owner._threads.pop(key, None), Qt.ConnectionType.QueuedConnection
    )
    thread.start()
    return thread


@dataclass
class _Running:
    """One in-flight job, held so neither half is collected while it runs."""

    thread: QThread
    worker: QObject


class BatchRunner(QObject):
    """A folder of captures, measured off the UI thread.

    Thin over :func:`deltachrome.batch.run_batch`, and deliberately so: everything
    that decides what a batch *is* -- that a failure never stops it, that records
    are written as they are made, that the target is declared rather than
    identified -- lives in the engine, where the command line reaches it too. What
    is here is the two things a window needs and a script does not.

    **Progress per capture, not per batch.** The signal carries how many have been
    done and how many there are, because on a folder of a hundred JPEG 2000 masters
    at sixteen seconds each the honest answer to "is this working" is a count.

    **Stopping.** ``stop`` is read by the engine between captures, so a run ends
    after the capture in flight rather than in the middle of one -- which keeps
    every record that exists complete, and is why the flag is polled rather than
    the thread being killed.
    """

    progress = Signal(int, int, str, object)  # done, total, capture name, BatchCapture
    finished = Signal(object)  # BatchResult
    failed = Signal(object)  # LoadFailure

    def __init__(self, *, captures, target_id: str, profile_id: str, into,
                 corners=None, reference_path=None, reference_library=None,
                 options=None, resume: bool = False) -> None:
        super().__init__()
        self.captures = list(captures)
        self.target_id = target_id
        self.profile_id = profile_id
        self.into = into
        self.corners = list(corners) if corners else None
        self.reference_path = reference_path
        self.reference_library = reference_library
        self.options = dict(options or {})
        self.resume = resume
        self._stop = False

    def stop(self) -> None:
        """Ask the run to finish after the capture it is on.

        Set from the window's thread and read from the worker's. A bare flag is
        enough: one writer, one reader, and a bool assignment a reader cannot see
        half of.
        """
        self._stop = True

    def run(self) -> None:
        from deltachrome.batch import run_batch

        try:
            result = run_batch(
                self.captures,
                target=self.target_id,
                profile=self.profile_id,
                into=self.into,
                corners=self.corners,
                reference_path=self.reference_path,
                reference_library=self.reference_library,
                resume=self.resume,
                on_progress=lambda done, total, path, capture: self.progress.emit(
                    done, total, path.name, capture
                ),
                should_stop=lambda: self._stop,
                **self.options,
            )
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            self.failed.emit(describe_analysis_failure(exc))
            return
        self.finished.emit(result)
