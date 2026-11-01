"""Command line interface.

Thin: it parses arguments, calls the engine, and prints. All logic lives in the
library so a GUI or web frontend can call exactly the same entry points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deltachrome import __version__
from deltachrome.inputs import parse_corners, parse_extent
from deltachrome.errors import DeltachromeError

EXIT_OK = 0
EXIT_FAIL = 1  # analysis ran and the capture failed the profile
EXIT_ERROR = 2  # analysis could not run
EXIT_INCOMPLETE = 3  # analysis ran but does not demonstrate conformance


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return EXIT_ERROR
    try:
        return int(args.handler(args))
    except DeltachromeError as exc:
        print(f"\nerror: {exc}\n", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        return EXIT_ERROR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deltachrome",
        description="Cultural heritage imaging QC — analyze test target captures "
        "against FADGI, Metamorfoze and ISO 19264.",
    )
    parser.add_argument("--version", action="version", version=f"deltachrome {__version__}")
    sub = parser.add_subparsers(dest="command")

    analyze = sub.add_parser("analyze", help="analyze one capture of a test target")
    analyze.add_argument("image", type=Path)
    analyze.add_argument("--identify", action="store_true",
                         help="work out which chart the capture is of, instead of "
                              "giving --target. Implies --detect. The run stops rather "
                              "than guessing where more than one target fits, and says "
                              "which they were")
    analyze.add_argument("--candidates", nargs="+", default=None, metavar="TARGET",
                         help="with --identify, choose only among these installed "
                              "target ids -- the charts you own. Explicit rather than "
                              "read from the desktop application's settings, so a "
                              "script names what it searched and reproduces anywhere")
    analyze.add_argument("--target", default=None,
                         help="installed target id, or a path to a target .json")
    analyze.add_argument("--reference", type=Path, default=None,
                         help="per-unit measurement file for this physical chart "
                              "(CGATS or CxF3)")
    analyze.add_argument("--reference-library", type=Path, default=None,
                         help="directory of reference files; the chart's printed code "
                              "is read from the capture and used to pick one")
    analyze.add_argument("--reference-table", default=None,
                         help="which table of a multi-table CGATS reference to use: a "
                              "1-based number, or a measurement condition such as M1. "
                              "Defaults to the table measured under M1 where the "
                              "file holds one, and to the first otherwise")
    analyze.add_argument("--no-nominal-fallback", action="store_true",
                         help="refuse to run without a per-unit reference file, rather "
                              "than falling back to the target's manufacturer nominal "
                              "values")
    analyze.add_argument("--no-verify-code", action="store_true",
                         help="skip cross-checking an explicit --reference against the "
                              "code printed on the chart")
    analyze.add_argument("--profile", required=True,
                         help="installed profile id, or a path to a profile .json")
    analyze.add_argument("--detect", action="store_true",
                         help="find the chart in the capture and place it automatically, "
                              "instead of giving --corners. The placement is refined onto "
                              "the chart's own printed features afterwards, and the run "
                              "stops rather than guessing if the chart cannot be found")
    analyze.add_argument("--detect-all", action="store_true",
                         help="read every chart in the frame, not just one: place each, "
                              "analyze each as its own run sharing the single decode, "
                              "and combine them into a session. For a board of targets "
                              "shot at several field positions. Implies --detect. With "
                              "--identify each chart is named separately, so a frame may "
                              "hold different models; a region holding a chart that "
                              "cannot be named is reported rather than dropped")
    analyze.add_argument("--save-session", type=Path, default=None,
                         help="with --detect-all, write the combined session record as "
                              "JSON. The individual runs are written by --save-run")
    analyze.add_argument("--corners", default=None,
                         help="chart corners in image pixels, in the TARGET's own "
                              "orientation: TL TR BR BL, where TL is the corner at "
                              "patch A1. A chart photographed rotated needs no special "
                              "handling -- give the corners in the target's order and "
                              "the homography absorbs the rotation. "
                              'Format: "x,y x,y x,y x,y"')
    analyze.add_argument("--refine-placement", action="store_true",
                         help="snap the given corners onto the chart's own printed "
                              "features before analyzing, and report how far they "
                              "moved. A placement can look right and still be far "
                              "enough out to stop a printed scale being readable")
    analyze.add_argument("--roi-inset", type=float, default=None,
                         help="fraction of each patch to sample (default 0.55)")
    analyze.add_argument("--fiducial-extent", default=None, metavar="WxH",
                         help="physical size in mm of the area this specimen's "
                              "fiducials enclose, e.g. 247.6x176.1. Use when the "
                              "target carries no printed scale; it describes the unit "
                              "you measured and does not transfer to another")
    analyze.add_argument("--percentile-method", default=None,
                         choices=["linear", "nearest_rank", "higher", "lower", "midpoint"],
                         help="override the profile's percentile interpolation")
    analyze.add_argument("--frequency-units", default=None,
                         choices=["lp/mm", "cycles/mm", "lp/inch", "cycles/inch",
                                  "cycles/px", "% of half-sampling freq"],
                         help="units for MTF50. Anything per mm or per inch needs a "
                              "sampling rate: a chart with a printed scale, or "
                              "--fiducial-extent. cycles/px always works")
    analyze.add_argument("--original-size", default=None, metavar="SIZE",
                         help="largest originals this system is being qualified for, "
                              "for standards whose thresholds band on it (Metamorfoze "
                              "illumination). A band or paper name (A2), a measurement "
                              "in mm (500x700), or the longest dimension (450). Without "
                              "it the metric is measured and reports the largest band "
                              "it clears, rather than being judged")
    analyze.add_argument("--save-run", type=Path, default=None,
                         help="write the full run record (raw per-patch data) as JSON")
    analyze.add_argument("--export", type=Path, action="append", default=[],
                         metavar="PATH",
                         help="write the report as .txt, .csv or .xml, chosen by the "
                              "extension. Repeatable. A capture holding several charts "
                              "is written as one file holding all of them")
    analyze.add_argument("--dump-overlay", type=Path, default=None,
                         help="write a PNG showing the sampled ROIs over the image")
    analyze.add_argument("--all-patches", action="store_true",
                         help="list every patch in the report, not just the worst")
    analyze.set_defaults(handler=_cmd_analyze)

    profile = sub.add_parser(
        "profile",
        help="build a camera input ICC profile from a capture of a colour chart, "
             "with ArgyllCMS's colprof",
    )
    profile.add_argument("image", type=Path,
                         help="the capture, usually with no embedded profile -- "
                              "device values, such as RawTherapee's Save Reference "
                              "Image")
    profile.add_argument("--out", type=Path, required=True,
                         help="where to write the .icc")
    profile.add_argument("--target", default=None,
                         help="installed target id, or a path to a target .json")
    profile.add_argument("--identify", action="store_true",
                         help="work out which chart it is, instead of --target")
    profile.add_argument("--reference", type=Path, default=None,
                         help="per-unit measurement file for this physical chart")
    profile.add_argument("--reference-library", type=Path, default=None,
                         help="directory of reference files, chosen by the chart's "
                              "printed code")
    profile.add_argument("--reference-table", default=None,
                         help="which table of a multi-table CGATS reference to use")
    profile.add_argument("--no-verify-code", action="store_true",
                         help="skip cross-checking --reference against the chart's "
                              "printed code")
    profile.add_argument("--corners", default=None,
                         help='chart corners in image pixels, TL TR BR BL: "x,y x,y x,y x,y"')
    profile.add_argument("--detect", action="store_true",
                         help="find and place the chart automatically")
    profile.add_argument("--refine-placement", action="store_true",
                         help="snap --corners onto the chart's printed features")
    profile.add_argument("--type", dest="profile_type", default="automatic",
                         choices=["automatic", "matrix", "shaper", "single-shaper",
                                  "gamma", "single-gamma", "xyz-table", "lab-table"],
                         help="what colprof fits (-a). automatic (default) tries the "
                              "candidates and keeps the one that best predicts "
                              "patches it was not built from")
    profile.add_argument("--white", default="absolute",
                         choices=["absolute", "chart", "chart-scaled"],
                         help="absolute (-ua, default): the chart reads its measured "
                              "values. chart: chart white is perfect white. "
                              "chart-scaled (-u): scaled so lighter values survive")
    profile.add_argument("--scale", type=float, default=None,
                         help="white scale (-u <scale>), with absolute or chart-scaled. "
                              "Without it the scale is computed; see --scale-for")
    profile.add_argument("--scale-for", default=None, choices=["white", "ramp", "none"],
                         help="what a computed scale aims at: white (default) puts the "
                              "patch whose reference is the lightest neutral on its "
                              "reference L*; ramp makes the "
                              "worst ΔL*2000 along the ramp as small as it can be; "
                              "none builds with no scale")
    profile.add_argument("--quality", default="automatic",
                         choices=["automatic", "low", "medium", "high"],
                         help="colprof's -q. automatic (default) is chosen with --type")
    profile.add_argument("--name", default=None,
                         help="the profile's description, as menus show it")
    profile.add_argument("--copyright", default="", help="the profile's copyright text")
    profile.add_argument("--exclude", nargs="+", default=[], metavar="PATCH",
                         help="patches to leave out of the fit. Flagged patches are "
                              "left out anyway")
    profile.add_argument("--restrict-primaries", action="store_true",
                         help="-R: keep white, black and primaries in range, for "
                              "software that rejects a negative primary")
    profile.add_argument("--merge-repeats", action="store_true",
                         help="read a colour the chart prints more than once as "
                              "one patch, its readings averaged")
    profile.add_argument("--smoothing", type=float, default=None, metavar="PERCENT",
                         help="-r: the readings' average deviation a table is "
                              "smoothed for (default 0.5). Table types only")
    profile.add_argument("--roi-inset", type=float, default=None,
                         help="fraction of each patch to sample (default 0.55)")
    profile.add_argument("--folds", type=int, default=5,
                         help="score each patch with a build that left it out, the "
                              "patches split into this many groups (default 5); 0 "
                              "skips it")
    profile.add_argument("--colprof", type=Path, default=None,
                         help="the colprof to run; otherwise DELTACHROME_COLPROF, "
                              "then the PATH")
    profile.add_argument("--write-ti3", type=Path, default=None,
                         help="also write the .ti3 colprof was given")
    profile.set_defaults(handler=_cmd_profile)

    session = sub.add_parser(
        "session",
        help="analyze several captures against one profile and combine them",
    )
    session.add_argument("file", type=Path, nargs="?",
                         help="session file: one profile, several captures")
    session.add_argument("--from-runs", type=Path, nargs="+", default=None,
                         metavar="RUN.json",
                         help="combine runs that were already analyzed and saved, "
                              "instead of re-measuring from a session file. Every "
                              "number a session reduces is in each record already, "
                              "so this costs no decode. They must share a profile")
    session.add_argument("--profile", default=None,
                         help="override the profile the session file names")
    session.add_argument("--original-size", default=None, metavar="SIZE",
                         help="largest originals this system is being qualified for "
                              "(see 'analyze --original-size'). Overrides the session "
                              "file's own value")
    session.add_argument("--save-session", type=Path, default=None,
                         help="write the combined session record as JSON")
    session.set_defaults(handler=_cmd_session)

    batch = sub.add_parser(
        "batch",
        help="analyze every capture in a folder against one guideline",
    )
    batch.add_argument("folder", type=Path,
                       help="folder of captures. Anything that is not an image is "
                            "passed over rather than reported as a failure")
    batch.add_argument("--target", required=True,
                       help="which chart is in every frame. Required, and "
                            "deliberately not optional: identification names a chart "
                            "from the capture, and on a frame that is mostly subject "
                            "it can name one confidently and wrongly -- which watched "
                            "is a bad suggestion and unattended is a hundred of them")
    batch.add_argument("--profile", required=True,
                       help="guideline every capture is judged against")
    batch.add_argument("--into", type=Path, required=True,
                       help="folder the records and the batch index are written to, "
                            "each as it is made")
    batch.add_argument("--seed-from", type=Path, default=None, metavar="RUN|IMAGE",
                       help="where to look, taken from a placement already made: a "
                            "SAVED RUN RECORD, whose corners somebody checked, or a "
                            "capture the chart can be found in on its own. The chart "
                            "is still found and refined in every frame -- this only "
                            "narrows the search, which is what makes a chart "
                            "occupying a sixth of its frame findable at all. Prefer "
                            "the run record: a chart that needs a seed is usually a "
                            "chart automatic whole-frame detection cannot place, so "
                            "seeding from an image only works where it is not needed")
    batch.add_argument("--corners", default=None,
                       help="the same, given directly as x,y;x,y;x,y;x,y (TL TR BR BL)")
    batch.add_argument("--recursive", action="store_true",
                       help="include captures in subfolders")
    batch.add_argument("--resume", action="store_true",
                       help="skip captures the index in --into already records as "
                            "measured, so a run that stopped at frame ninety costs "
                            "ten frames rather than a hundred")
    batch.add_argument("--reference", type=Path, default=None,
                       help="one measurement file for the whole batch. Legitimate "
                            "where the batch is one physical chart photographed many "
                            "times, which is the ordinary case")
    batch.add_argument("--reference-library", type=Path, default=None,
                       help="pair each capture with its own file by the code printed "
                            "on the chart, instead of naming one")
    batch.add_argument("--roi-inset", type=float, default=None,
                       help="fraction of each patch sampled")
    batch.add_argument("--percentile-method", default=None,
                       choices=["linear", "nearest_rank"])
    batch.add_argument("--frequency-units", default=None,
                       help="unit MTF50 is reported in (see 'analyze')")
    batch.add_argument("--original-size", default=None, metavar="SIZE",
                       help="largest originals this system is being qualified for")
    batch.add_argument("--fiducial-extent", default=None, metavar="WxH",
                       help="physical size of this unit's fiducial quad, in mm")
    batch.set_defaults(handler=_cmd_batch)

    derive = sub.add_parser(
        "derive",
        help="build a local guideline from a standard profile and save it",
    )
    derive.add_argument("base", help="profile id or path to derive from")
    derive.add_argument("new_id", help="id for the derived profile")
    derive.add_argument("--set", action="append", default=[], metavar="METRIC=CONDITION",
                        help="override a threshold, e.g. "
                             "--set sampling_frequency='value >= 2000'. Repeatable")
    derive.add_argument("--omit", action="append", default=[], metavar="METRIC",
                        help="drop a metric entirely. Repeatable")
    derive.add_argument("--name", default=None, help="display name for the result")
    derive.add_argument("--into", type=Path, default=None,
                        help="directory to save into (default: the user profile "
                             "directory)")
    derive.add_argument("--dry-run", action="store_true",
                        help="show what would change without saving")
    derive.set_defaults(handler=_cmd_derive)

    targets = sub.add_parser("targets", help="list installed target definitions")
    targets.set_defaults(handler=_cmd_targets)

    profiles = sub.add_parser("profiles", help="list and filter installed analysis profiles")
    profiles.add_argument("--body", default=None,
                          help="standards body, e.g. FADGI")
    profiles.add_argument("--material", default=None,
                          help="material category substring, e.g. 'transparencies'")
    profiles.add_argument("--star", type=int, default=None, help="star / performance level")
    profiles.add_argument("--edition", default=None,
                          help="document edition substring, e.g. '3rd'")
    profiles.add_argument("--metrics", action="store_true",
                          help="list each profile's metrics and thresholds")
    profiles.set_defaults(handler=_cmd_profiles)

    metrics = sub.add_parser("metrics", help="list registered metric plugins")
    metrics.set_defaults(handler=_cmd_metrics)

    inspect_ref = sub.add_parser("inspect-reference", help="parse and summarize a CGATS or CxF3 reference file")
    inspect_ref.add_argument("path", type=Path)
    inspect_ref.add_argument("--target", default=None,
                             help="also report coverage against this target's patch ids")
    inspect_ref.set_defaults(handler=_cmd_inspect_reference)

    references = sub.add_parser("references", help="list or edit a reference library")
    references.add_argument("--library", type=Path, default=None,
                            help="the library folder; the application's reference bank "
                                 "by default")
    references.add_argument("--bind", metavar="CODE=FILENAME", default=None,
                            help="bind a chart code to a reference file in the library")
    references.set_defaults(handler=_cmd_references)

    identify = sub.add_parser("identify",
                              help="read the chart's printed code and pair it with a reference")
    identify.add_argument("image", type=Path)
    identify.add_argument("--library", type=Path, default=None)
    identify.add_argument("--no-tiles", action="store_true",
                          help="only run the fast downsampled pass")
    identify.set_defaults(handler=_cmd_identify)

    export = sub.add_parser(
        "export",
        help="write a saved run or capture record as .txt, .csv or .xml",
        description="Write a saved record out for somebody not running Deltachrome. "
                    "The format is chosen by each output's extension; the record is "
                    "read, never re-measured.",
    )
    export.add_argument("record", type=Path, help="a run or capture record (.json)")
    export.add_argument("outputs", type=Path, nargs="+", metavar="OUTPUT",
                        help="one or more files to write: .txt, .csv or .xml")
    export.set_defaults(handler=_cmd_export)

    inspect_icc = sub.add_parser("inspect-icc", help="report an image's embedded ICC profile")
    inspect_icc.add_argument("path", type=Path)
    inspect_icc.set_defaults(handler=_cmd_inspect_icc)

    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    from deltachrome.imaging.loader import load_image
    from deltachrome.imaging.sampling import DEFAULT_ROI_INSET
    from deltachrome.pipeline import analyze
    from deltachrome.reporting.console import render
    from deltachrome.reporting.export import format_for
    from deltachrome.targets.registry import load_target

    # Before the decode rather than after it: a mistyped extension found once the
    # analysis has finished costs the whole run.
    for path in args.export:
        format_for(path)
    if args.identify and args.target:
        raise SystemExit(
            "give --target or --identify, not both: it would be unsaid which one the "
            "run was judged as, and a run record has to say."
        )
    if not args.identify and not args.target:
        raise SystemExit("give --target, or --identify to work out which chart it is.")
    if args.candidates and not args.identify:
        raise SystemExit(
            "--candidates narrows what --identify chooses among, and means nothing "
            "without it."
        )
    if args.candidates:
        from deltachrome.imaging.detect import identifiable_targets

        unknown = sorted(set(args.candidates) - set(identifiable_targets()))
        if unknown:
            raise SystemExit(
                f"--candidates names {', '.join(unknown)}, which identification "
                f"cannot name. Choose from: {', '.join(identifiable_targets())}."
            )
    if args.identify and args.corners:
        raise SystemExit(
            "--identify finds the chart as well as naming it, so it cannot be given "
            "with --corners."
        )
    if args.detect_all and args.corners:
        raise SystemExit(
            "give --corners or --detect-all, not both: --corners places one chart, "
            "and it would be unsaid which of the charts found it belonged to."
        )
    if args.save_session and not args.detect_all:
        raise SystemExit(
            "--save-session writes the record combining several charts read from one "
            "frame, which is what --detect-all produces. One chart is one run; save it "
            "with --save-run."
        )
    # Identifying a chart places it on the way past: the candidate that won did so
    # partly by its regions landing on patches, so the placement is already made and
    # asking for it again would be a second search for the same answer.
    args.detect = args.detect or args.identify or args.detect_all
    if args.detect and args.corners:
        raise SystemExit(
            "give --corners or --detect, not both: it would be unsaid which one the "
            "numbers came from, and a run record has to say."
        )
    if not args.detect and not args.corners:
        raise SystemExit(
            "give --corners to place the chart by hand, or --detect to find it. "
            "Neither is a default here: the desktop application shows an automatic "
            "placement for checking before anything is measured and a scripted run "
            "has no such moment, so which one produced the geometry is asked for "
            "rather than assumed."
        )
    corners = _parse_corners(args.corners) if args.corners else None
    # Target first: an unknown id is a typo, and a typo should not cost the
    # decode of a 577 MB master before it is reported.
    target = load_target(args.target) if args.target else None
    # Decoded exactly once and handed to both the code scan and the analysis.
    # These were separate loads until the resolver moved into the engine, which
    # meant every run with code verification on -- the default -- decoded the
    # master twice.
    image = load_image(args.image)
    identified: list[str] = []
    was_identified = target is None and not args.detect_all
    placed = "by_hand"
    if was_identified:
        target, corners, identified = _identify(image, args.candidates)
        if corners is not None:
            placed = "automatic"
    if target is None:
        # --identify with --detect-all: each chart is named and paired separately,
        # since a frame may hold two models needing two different reference files.
        reference_path, code_notes, code_warnings = None, [], []
    else:
        reference_path, code_notes, code_warnings = _resolve_reference(
            args, image=image, target=target
        )
    for note in code_notes:
        print(note)
    code_warnings = list(code_warnings) + identified

    if args.detect_all:
        return _analyze_every_chart(args, image, target, reference_path, code_warnings)

    if corners is None:
        corners, detect_notes, code_warnings = _detect_corners(
            image, target, code_warnings
        )
        placed = "automatic"
        for note in detect_notes:
            print(note)
    elif args.refine_placement:
        corners, refine_notes, code_warnings = _refine_corners(
            image, corners, target, code_warnings
        )
        for note in refine_notes:
            print(note)
    result = analyze(
        image,
        # Already loaded above. resolve_roles builds a new definition rather than
        # mutating this one, so reusing it is safe.
        target=target,
        reference_path=reference_path,
        profile=args.profile,
        corners=corners,
        roi_inset=args.roi_inset if args.roi_inset is not None else DEFAULT_ROI_INSET,
        percentile_method=args.percentile_method,
        frequency_units=args.frequency_units,
        original_size=args.original_size,
        extra_warnings=code_warnings,
        fiducial_extent_mm=_parse_extent(args.fiducial_extent),
        allow_nominal_reference=not args.no_nominal_fallback,
        reference_table=args.reference_table,
        placed=placed,
        identified=was_identified,
        searched_targets=args.candidates if was_identified else None,
    )

    print(render(result, show_all_patches=args.all_patches))

    if args.save_run:
        path = result.run.save(args.save_run)
        print(f"\nrun record written to {path}")
    _write_exports(args.export, result.run.to_dict())

    if args.dump_overlay:
        import cv2

        from deltachrome.imaging.sampling import render_overlay

        args.dump_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = render_overlay(result.image, result.placement, result.sampling)
        cv2.imwrite(str(args.dump_overlay), overlay)
        print(f"overlay written to {args.dump_overlay}")

        if result.edges:
            from deltachrome.imaging.edges import render_edge_overlay

            path = args.dump_overlay.with_name(
                args.dump_overlay.stem + "_edges" + args.dump_overlay.suffix
            )
            cv2.imwrite(
                str(path),
                render_edge_overlay(
                    result.image, result.placement, result.target, result.edges
                ),
            )
            print(f"edge overlay written to {path}")

    return {"PASS": EXIT_OK, "MEASURED": EXIT_OK, "FAIL": EXIT_FAIL}.get(
        result.overall, EXIT_INCOMPLETE
    )


def _cmd_profile(args: argparse.Namespace) -> int:
    """Build a camera profile: find the chart, sample it, run colprof, score it."""
    from deltachrome.errors import ProfilingError
    from deltachrome.imaging.loader import load_image
    from deltachrome.imaging.sampling import DEFAULT_ROI_INSET
    from deltachrome.profiling import (
        EXCLUDED_FLAGGED,
        TABLE_TYPES,
        ColprofOptions,
        build_automatic,
        build_profile,
        check_reference,
        find_colprof,
        format_ti3,
        gather_patches,
        merge_repeats,
    )
    from deltachrome.profiling.encoding import detect_encoding
    from deltachrome.reference import read_reference
    from deltachrome.results.models import file_fingerprint
    from deltachrome.targets.registry import load_target

    if args.identify and args.target:
        raise SystemExit("give --target or --identify, not both.")
    if not args.identify and not args.target:
        raise SystemExit("give --target, or --identify to work out which chart it is.")
    if args.corners and (args.detect or args.identify):
        raise SystemExit("give --corners or --detect, not both.")
    if not args.corners and not (args.detect or args.identify):
        raise SystemExit("give --corners to place the chart by hand, or --detect to find it.")
    if args.out.suffix.lower() not in (".icc", ".icm"):
        raise SystemExit("--out should name an .icc file.")
    if args.scale is not None and args.scale_for not in (None, "none"):
        raise SystemExit("give --scale or --scale-for, not both.")
    # Computed by default, the way the desktop application's Scale reads
    # Automatic -- except where colprof cannot take a scale at all.
    scale_for = args.scale_for or ("white" if args.white != "chart" else "none")
    if args.scale is not None or scale_for == "none":
        scale_for = None

    target = load_target(args.target) if args.target else None
    # Before the decode, so a missing colprof or a bad option costs nothing.
    colprof = find_colprof(args.colprof)
    automatic = "automatic" in (args.profile_type, args.quality)
    if args.smoothing is not None and args.profile_type not in TABLE_TYPES:
        raise SystemExit("--smoothing applies to a table profile: give --type "
                         "xyz-table or lab-table.")
    if automatic and args.folds < 2:
        raise SystemExit("automatic chooses by held-out error, which needs --folds 2 or more.")
    # Placeholders where Automatic will choose; ColprofOptions holds only real values.
    options = ColprofOptions(
        profile_type="shaper" if args.profile_type == "automatic" else args.profile_type,
        white=args.white,
        scale=args.scale,
        quality="medium" if args.quality == "automatic" else args.quality,
        description=args.name or args.out.stem,
        copyright=args.copyright,
        restrict_primaries=args.restrict_primaries,
        smoothing=args.smoothing,
    )

    image = load_image(args.image, profile_required=False)
    print(image.summary())
    # What placement had to say is printed as it happens; with no run record to
    # carry it into, repeating it at the end would only say it twice.
    warnings: list[str] = []
    corners = _parse_corners(args.corners) if args.corners else None
    if target is None:
        target, corners, _ = _identify(image)
    if corners is None:
        corners, notes, _ = _detect_corners(image, target, [])
        for note in notes:
            print(note)
    elif args.refine_placement:
        corners, notes, _ = _refine_corners(image, corners, target, [])
        for note in notes:
            print(note)
    options.transmissive = "transmissive" in (target.target_type or "")

    # The pairing's own warnings are among its notes, printed here; they are not
    # carried on, there being no run record to carry them into.
    reference_path, code_notes, _ = _resolve_reference(args, image=image, target=target)
    for note in code_notes:
        print(note)
    if reference_path is None:
        reference_path = target.nominal_reference_path
    if reference_path is None:
        raise ProfilingError(
            f"No reference file was given, and {target.display_name} ships none. "
            "A profile is built against the chart's own measurement."
        )
    reference = read_reference(reference_path, table=args.reference_table)
    warnings.extend(reference.warnings)
    warnings.extend(check_reference(reference, Path(reference_path).name))

    data = gather_patches(
        image, target, reference, corners,
        exclude=args.exclude,
        roi_inset=args.roi_inset if args.roi_inset is not None else DEFAULT_ROI_INSET,
    )
    warnings.extend(data.warnings)
    gathered = data
    if args.merge_repeats:
        data = merge_repeats(data)
    fingerprint = file_fingerprint(args.image) or {}
    keywords = {
        "target": target.target_id,
        "reference": Path(reference_path).name,
        "capture": args.image.name,
        "capture_digest": fingerprint.get("digest", ""),
    }
    if args.merge_repeats:
        keywords["merged_repeats"] = "yes"
    if args.write_ti3:
        args.write_ti3.write_text(format_ti3(data, keywords), encoding="ascii")
        print(f"  .ti3 written to {args.write_ti3}")

    print(f"\n  {target.display_name}, against {Path(reference_path).name}")
    flagged = [pid for pid, (_, why) in data.excluded.items() if why == EXCLUDED_FLAGGED]
    print(f"  {len(gathered.used)} of {len(gathered.all_patches)} patches used"
          + (f"; flagged: {', '.join(flagged)}" if flagged else ""))
    if data is not gathered:
        families = sum(1 for patch in data.used if patch.members)
        print(f"  repeats merged: {len(data.used)} readings, {families} of them merged")
    encoding = detect_encoding(gathered)
    if encoding is not None:
        print(f"  {encoding.label}")

    if automatic:
        print("  choosing the type and quality by validation error...")
        built = build_automatic(
            data, options, colprof=colprof, keywords=keywords, scale_for=scale_for,
            folds=args.folds,
            types=None if args.profile_type == "automatic" else [args.profile_type],
            qualities=None if args.quality == "automatic" else [args.quality],
        )
        print("\n  validation     mean     90th    scale")
        won = (built.options.profile_type, built.options.quality)
        for trial in built.trials:
            mark = "*" if (trial.profile_type, trial.quality) == won else " "
            scale = f"{trial.scale:.4f}" if trial.scale is not None else "-"
            print(f"  {mark} {trial.profile_type + ' ' + trial.quality:<22}"
                  f"{trial.held_out_mean:5.2f} {trial.held_out_p90:8.2f} {scale:>8}")
    else:
        built = build_profile(
            data, options, colprof=colprof, keywords=keywords, scale_for=scale_for,
            folds=args.folds,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(built.icc)
    print(f"  colprof {' '.join(built.run.arguments)}")
    if built.scale_aim:
        print(f"  scale {built.scale:.5g}, computed for {built.scale_aim} "
              f"({built.scale_basis})")

    print("\n  ΔE2000          mean     90th      max")
    for label, summary in (("Built From", built.fit), ("Validation", built.held_out),
                           ("Excluded", built.excluded)):
        if summary is not None:
            print(f"  {label:<12} {summary.mean:8.2f} {summary.p90:8.2f} {summary.max:8.2f}"
                  f"   ({summary.count})")
    for warning in warnings + built.run.warnings:
        print(f"  WARNING: {warning}")
    print(f"\nprofile written to {args.out}")
    return EXIT_OK


def _charts_in_frame(args, image, target):
    """Every chart to measure, as (target, placement), and what was left unmeasured.

    Two ways in, and they differ only in where the target comes from. With
    ``--target`` every chart in the frame is that model, which is the QA-62 board:
    four cards of one kind. With ``--identify`` each region is asked what it holds,
    so a frame may yield a Device Level Target and a ColorChecker and a scale, each
    with its own target and its own reference.

    **A region that held a chart but could not be named is returned as skipped, not
    dropped.** Identification declines rather than coin-tossing where two targets fit
    equally, and a run that quietly measured three of four charts would be worse than
    one that measured three and said so -- the fourth would look like it was not
    there. The reasons travel into every run's warnings as well as to the terminal,
    because the record is what a reader has a year later.
    """
    from deltachrome.imaging.detect import detect_placements, identify_targets
    from deltachrome.targets.registry import load_target

    if not args.identify:
        return [
            (target, detection)
            for detection in detect_placements(image, target) if detection.found
        ], []

    charts, skipped = [], []
    for result in identify_targets(image, args.candidates):
        if result.found and result.detection is not None and result.detection.found:
            charts.append((load_target(result.target_id), result.detection))
        else:
            skipped.append(result.reason)

    # A per-unit measurement file describes one physical chart. Handing the same one
    # to several charts of different models is meaningless, and to several units of
    # one model it is wrong in a way nothing downstream could notice -- so it is
    # refused here rather than reported afterwards.
    if args.reference and len(charts) > 1:
        raise SystemExit(
            f"--reference names one physical chart's measurement file, and "
            f"{len(charts)} charts were found in this capture. Use "
            "--reference-library, which pairs each chart with its own file by the "
            "code printed on it, or analyze the charts one at a time with --corners."
        )
    return charts, skipped


def _analyze_every_chart(args, image, target, reference_path, warnings) -> int:
    """Read every chart in one frame: several runs sharing one decode, then a session.

    **Deliberately not one run with several charts in it.** A run is one capture,
    one target, one placement, one verdict, and every record in the library is that
    shape; widening it to hold a list of placements would change the shape of every
    record ever saved to accommodate an arrangement of charts. So a frame holding
    four cards is four ordinary runs, each saveable, reopenable and re-scorable on
    its own, and the question the board is actually asking -- did the *system* pass,
    across all four field positions -- is the question a session already answers.

    What is shared is the decode, which is the only expensive part: 10 seconds for
    the QA-62 board's 101-megapixel JPEG 2000 master, paid once rather than four
    times.
    """
    from deltachrome.errors import GeometryError
    from deltachrome.imaging.detect import (
        detect_placements, identify_targets, unidentified_regions_note,
    )
    from deltachrome.imaging.sampling import DEFAULT_ROI_INSET
    from deltachrome.pipeline import analyze
    from deltachrome.reporting.console import render, render_session
    from deltachrome.session import session_from_results
    from deltachrome.targets.registry import load_target

    charts, skipped = _charts_in_frame(args, image, target)
    if not charts:
        raise GeometryError(
            "no chart could be placed automatically in this capture. Pass --corners "
            "to place one by hand."
        )
    print(f"  {len(charts)} chart(s) placed in this frame.")
    for reason in skipped:
        print(f"  skipped: {reason}")

    runs = []
    for number, (chart_target, detection) in enumerate(charts, start=1):
        print(f"\n  chart {number}: {chart_target.display_name} -- {detection.reason}")
        for note in detection.notes:
            print(f"    {note}")
        # Resolved per chart, because the charts may be different models and a
        # per-unit file describes one physical chart. _charts_in_frame has already
        # refused the ambiguous combination.
        chart_reference = reference_path
        if args.identify:
            chart_reference, notes, _warned = _resolve_reference(
                args, image=image, target=chart_target
            )
            for note in notes:
                print(f"    {note}")
        runs.append(analyze(
            image,
            target=chart_target,
            reference_path=chart_reference,
            profile=args.profile,
            corners=detection.corners,
            roi_inset=args.roi_inset if args.roi_inset is not None else DEFAULT_ROI_INSET,
            percentile_method=args.percentile_method,
            frequency_units=args.frequency_units,
            original_size=args.original_size,
            placed="automatic",
            identified=bool(args.identify),
            searched_targets=args.candidates if args.identify else None,
            extra_warnings=list(warnings) + list(detection.notes) + [
                note for note in [unidentified_regions_note(len(skipped))] if note
            ],
            fiducial_extent_mm=_parse_extent(args.fiducial_extent),
            allow_nominal_reference=not args.no_nominal_fallback,
            reference_table=args.reference_table,
        ))
        print(f"\n  ---- chart {number} of {len(charts)} ----")
        print(render(runs[-1], show_all_patches=args.all_patches))
        if args.save_run:
            # Numbered per chart. One name for four records would leave three of them
            # unwritten, which is the sort of loss a run record exists to prevent.
            path = runs[-1].run.save(
                args.save_run.with_name(f"{args.save_run.stem}_{number}{args.save_run.suffix}")
            )
            print(f"\nrun record written to {path}")

    session = session_from_results(
        runs,
        labels=[f"chart {n} ({t.target_id})" for n, (t, _d) in enumerate(charts, 1)],
    )
    print(render_session(session))
    if args.save_session:
        import json as _json

        args.save_session.parent.mkdir(parents=True, exist_ok=True)
        args.save_session.write_text(
            _json.dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nsession record written to {args.save_session}")
    if args.export:
        from deltachrome.session import capture_record

        _write_exports(args.export, capture_record(
            runs, labels=[f"Target {n}" for n in range(1, len(runs) + 1)],
        ))

    return {"PASS": EXIT_OK, "MEASURED": EXIT_OK, "FAIL": EXIT_FAIL}.get(
        session.overall, EXIT_INCOMPLETE
    )


def _write_exports(paths, record) -> None:
    """Every ``--export``, each in the format its extension names.

    Every extension is checked before anything is written, so a typo in the
    third path does not leave the first two written and the run reporting an
    error.
    """
    if not paths:
        return
    from deltachrome.reporting.export import export, format_for

    for path in paths:
        format_for(path)
    for path in paths:
        print(f"\nreport exported to {export(record, path)}")


def _cmd_export(args: argparse.Namespace) -> int:
    import json

    from deltachrome.reporting.export import ExportError, format_for

    for path in args.outputs:
        format_for(path)
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExportError(f"{args.record} could not be read: {exc}") from exc
    # A run record has a run id and a capture record has charts. A session record
    # has metrics too, and neither of these: it combines several captures and is
    # not a report of any one of them.
    if not isinstance(record, dict) or not ("run_id" in record or "charts" in record):
        raise ExportError(
            f"{args.record.name} is not a run or capture record. A session record "
            "combines several captures and is not exported here."
        )
    _write_exports(args.outputs, record)
    return EXIT_OK


def _identify(image, candidates=None):
    """Work out which chart the capture holds, or stop.

    Stopping is the whole point of checking the return value rather than taking the
    best-ranked candidate. Naming the wrong chart is worse than naming none: every
    number that follows is measured against another chart's aim values, and the run
    reports a full set of them. So where more than one target fits, both are named
    and the run ends -- and that it was identified goes into the record, because a
    target nobody chose has to be visible in the archive: as ``geometry.identified``,
    not as a warning, since nothing about it is wrong.
    """
    from deltachrome.imaging.detect import identify_target
    from deltachrome.targets.registry import load_target

    print("  Working out which chart this is...")
    result = identify_target(image, candidates)
    if not result.found:
        raise SystemExit(
            f"the chart could not be identified: {result.reason}\n"
            "Pass --target to name it."
        )
    print(f"  {result.reason}")
    for note in result.notes:
        print(f"    {note}")
    detection = result.detection
    return (
        load_target(result.target_id),
        list(detection.corners) if detection is not None and detection.found else None,
        list(detection.notes if detection is not None else []),
    )


def _detect_corners(image, target, warnings):
    """Place the chart automatically, or stop.

    Refusing is the whole point of the return value being checked here rather
    than a placement being invented. A wrong automatic placement is worse than
    none: it samples every ROI from the wrong pixels and returns a full set of
    plausible numbers, which is the one failure this project has consistently
    treated as the serious one. So a refusal ends the run with the finder's own
    sentence and the instruction to place the chart by hand, and a detection that
    succeeded is recorded in the run's ``geometry.placed`` as what produced the
    geometry -- because a placement nobody chose has to be visible in the record.
    """
    from deltachrome.errors import GeometryError
    from deltachrome.imaging.detect import detect_placement

    detection = detect_placement(image, target)
    if not detection.found:
        # A GeometryError rather than a bare exit, so this leaves with
        # ``EXIT_ERROR`` -- "the analysis could not run" -- and not with
        # ``EXIT_FAIL``, which means the capture was measured and did not meet the
        # standard. A script batching a morning's captures has to be able to tell
        # a chart that failed from a chart that was never placed.
        raise GeometryError(
            f"the chart could not be placed automatically. {detection.reason} "
            "Pass --corners to place it by hand."
        )
    notes = [f"  {detection.reason}"] + [f"    {note}" for note in detection.notes]
    return (
        detection.corners,
        notes,
        list(warnings) + list(detection.notes),
    )


def _refine_corners(image, corners, target, warnings):
    """Apply placement refinement, reporting the outcome either way.

    Never silent. A refinement that moved the corners changes every measurement
    that follows, and one that refused leaves the operator with the placement
    they gave -- both are facts about the run, so both are printed and the
    outcome is carried into the record's warnings.
    """
    from deltachrome.imaging.geometry import place_chart
    from deltachrome.imaging.refine import refine_placement

    placement = place_chart(
        corners,
        image_size=(image.width, image.height),
        corner_positions_relative=target.fiducials.corner_positions or None,
    )
    result = refine_placement(image, placement, target)
    if not result.applied:
        message = f"Placement was not refined: {result.reason}"
        return corners, [f"  {message}"], list(warnings) + [message]

    notes = [f"  placement refined: {result.reason}"]
    notes += [f"    {note}" for note in result.notes]
    return (
        result.corners,
        notes,
        list(warnings) + [
            "The corners were refined onto the chart's printed features before "
            f"analysis. {result.reason} They agree with the fitted placement to "
            f"{result.residual_px:.2f} px RMS."
        ],
    )


def _cmd_batch(args: argparse.Namespace) -> int:
    """Run a folder, reporting each capture as it finishes.

    The exit code answers the question a script asks: did anything fail, and did
    everything get measured. A batch where every capture passed but three could not
    be read is not a pass, and it is not a failure either -- it is incomplete, and
    that is a third answer rather than a rounding of the other two.
    """
    from deltachrome.batch import captures_in, run_batch
    from deltachrome.imaging.detect import detect_placement
    from deltachrome.imaging.loader import load_image
    from deltachrome.imaging.sampling import DEFAULT_ROI_INSET
    from deltachrome.targets.registry import load_target

    if args.corners and args.seed_from:
        raise SystemExit(
            "give --corners or --seed-from, not both: they are two ways of saying "
            "where to look and it would be unsaid which one the run used."
        )
    if args.reference and args.reference_library:
        raise SystemExit(
            "give --reference or --reference-library, not both: one names a file for "
            "the whole batch and the other pairs each capture with its own."
        )

    target = load_target(args.target)
    captures = captures_in(args.folder, recursive=args.recursive)
    if not captures:
        raise SystemExit(f"no captures found in {args.folder}.")

    corners = _parse_corners(args.corners) if args.corners else None
    if args.seed_from:
        corners, why = _seed_corners(args.seed_from, target)
        print(f"seed placement from {args.seed_from.name}: {why}")

    print(f"{len(captures)} capture(s) in {args.folder}, "
          f"{target.display_name} against {args.profile}")

    def progress(number, total, path, capture) -> None:
        if capture is None:
            return
        mark = capture.overall if capture.measured else "NOT MEASURED"
        print(f"  [{number}/{total}] {path.name}: {mark}"
              + (f" -- {capture.reason}" if not capture.measured else ""))

    result = run_batch(
        captures, target=target, profile=args.profile, into=args.into,
        corners=corners, reference_path=args.reference,
        reference_library=args.reference_library, resume=args.resume,
        on_progress=progress,
        roi_inset=args.roi_inset if args.roi_inset is not None else DEFAULT_ROI_INSET,
        percentile_method=args.percentile_method,
        frequency_units=args.frequency_units,
        original_size=args.original_size,
        fiducial_extent_mm=_parse_extent(args.fiducial_extent),
    )

    print(f"\n{result.summary_line()}")
    if not result.conformance_claim and result.measured:
        print("  Advisory: not every capture can be quoted as a conformance test.")
    unusual = result.unusual_captures()
    if unusual:
        print(f"\n  UNLIKE THE REST OF THE BATCH ({len(unusual)} capture(s))")
        print("  A reading can meet its requirement and still be unlike the rest of")
        print("  its own batch, which is the one thing a batch says that its")
        print("  captures do not.")
        for name, found in unusual:
            print(f"    {name}")
            for outlier in found:
                print(f"      {outlier.metric_id} reads {outlier.value:.3f} against a "
                      f"batch median of {outlier.median:.3f}, {outlier.sigmas:.1f} "
                      f"sigma {outlier.direction}")
    print(f"\nrecords written to {args.into}")

    if result.failed:
        return EXIT_FAIL
    if result.refused or result.stopped_early or not result.conformance_claim:
        return EXIT_INCOMPLETE
    return EXIT_OK


def _seed_corners(path: Path, target) -> tuple[list, str]:
    """Where to look, from a run record's checked placement or from a capture.

    **The record is the better source and usually the only one that works.** A
    chart needing a seed at all is a chart the whole-frame search cannot place --
    that is what the seed is for -- so detecting one on a capture succeeds exactly
    where it was not needed. A saved run carries four corners somebody looked at
    before pressing Run, which is a stronger warrant than any search.
    """
    from deltachrome.imaging.detect import detect_placement, detect_placements
    from deltachrome.imaging.loader import load_image

    if path.suffix.lower() == ".json":
        import json as _json

        try:
            record = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise SystemExit(f"{path} could not be read as a run record: {exc}")
        charts = record.get("charts")
        if isinstance(charts, list) and charts:
            record = charts[0]
        raw = (record.get("geometry") or {}).get("corners_px")
        if not isinstance(raw, list) or len(raw) != 4:
            raise SystemExit(
                f"{path.name} is a record with no placement in it, so it cannot say "
                "where to look."
            )
        return (
            [(float(x), float(y)) for x, y in raw],
            "four corners from a saved run",
        )

    image = load_image(path)
    # Every chart in the frame, not the best single answer. On a board of four
    # QA-62 cards the single search returns a quad made of one crosshair from each
    # card -- a valid quadrilateral spanning the whole board, whose own grey ramp
    # reads -0.47 where a card reads +1.00 -- and a batch seeded from it measured
    # that quad in every frame and reported confident verdicts on nothing. Which
    # of several charts a batch should follow is the operator's to say.
    placements = [d for d in detect_placements(image, target) if d.found]
    if len(placements) > 1:
        raise SystemExit(
            f"{path.name} holds {len(placements)} charts of this model, and a batch "
            "measures one in each frame -- so a capture cannot say which to follow. "
            "Place the one you want in the desktop application, save that run, and "
            "pass the record here instead."
        )
    found = placements[0] if placements else detect_placement(image, target)
    if not found.found:
        raise SystemExit(
            f"the chart could not be placed in {path.name}, so it cannot say where "
            f"to look in the others: {found.reason}\n"
            "A chart that needs a seed is usually one whole-frame detection cannot "
            "place. Place it once in the desktop application, save that run, and "
            "pass the record here instead."
        )
    return found.corners, found.reason


def _cmd_derive(args: argparse.Namespace) -> int:
    from deltachrome.errors import ProfileError
    from deltachrome.profiles.schema import load_profile, user_profile_dir

    thresholds = {}
    for item in args.set:
        metric_id, sep, condition = item.partition("=")
        if not sep or not condition.strip():
            raise ProfileError(
                f"--set expects METRIC=CONDITION; got {item!r}. For example: "
                "--set sampling_frequency='value >= 2000'"
            )
        thresholds[metric_id.strip()] = condition.strip()

    base = load_profile(args.base)
    derived = base.derive(
        args.new_id, thresholds=thresholds, omit=args.omit, display_name=args.name
    )

    print(f"{derived.display_name}")
    print(f"  derived from {base.profile_id}")
    for entry in derived.departures:
        if entry["change"] == "omitted":
            print(f"    omitted   {entry['metric_id']}")
        else:
            print(f"    {entry['direction']:<9} {entry['metric_id']}: "
                  f"{entry['from']} -> {entry['to']}")
    print(
        "\n  A run against this is not a conformance test against the standard,\n"
        "  whichever way a threshold moved: it is a local guideline. It is not\n"
        "  thereby advisory -- on a chart that was itself measured the reading\n"
        "  stands, and the verdict says which guideline it is a verdict about."
    )
    if args.dry_run:
        print("\n  (dry run -- nothing written)")
        return EXIT_OK

    path = derived.save(args.into)
    print(f"\nsaved to {path}")
    if args.into is None:
        print(f"callable by id as {derived.profile_id!r} "
              f"(user profiles live in {user_profile_dir()})")
    return EXIT_OK


def _cmd_session(args: argparse.Namespace) -> int:
    import json as _json

    from deltachrome.reporting.console import render_session
    from deltachrome.session import (
        analyze_session,
        load_session_file,
        session_from_records,
    )

    from deltachrome.errors import ProfileError

    if args.from_runs:
        if args.file:
            raise ProfileError(
                "Pass either a session file or --from-runs, not both. A session file "
                "describes captures to measure; --from-runs combines measurements "
                "already made."
            )
        records = [_json.loads(path.read_text(encoding="utf-8")) for path in args.from_runs]
        result = session_from_records(records, profile=args.profile)
    else:
        if not args.file:
            raise ProfileError(
                "Give a session file, or --from-runs with saved run records."
            )
        captures, profile_id, settings = load_session_file(args.file)
        if args.original_size:
            settings["original_size"] = args.original_size
        result = analyze_session(captures, profile=args.profile or profile_id, **settings)
    print(render_session(result))

    if args.save_session:
        args.save_session.parent.mkdir(parents=True, exist_ok=True)
        args.save_session.write_text(
            _json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nsession record written to {args.save_session}")

    return {"PASS": EXIT_OK, "MEASURED": EXIT_OK, "FAIL": EXIT_FAIL}.get(
        result.overall, EXIT_INCOMPLETE
    )


def _cmd_targets(_args: argparse.Namespace) -> int:
    from deltachrome.targets.registry import available_targets, load_target

    for target_id in available_targets():
        target = load_target(target_id)
        print(f"{target_id}\n    {target.display_name} — {len(target.patches)} patches")
        print(f"    capabilities: {', '.join(sorted(target.capabilities))}")
        if target.not_capable_of:
            print(f"    cannot measure: {', '.join(sorted(target.not_capable_of))}")
    return EXIT_OK


def _cmd_profiles(args: argparse.Namespace) -> int:
    """List profiles, filtered by standards body, edition, material and level.

    A standard is a three-way choice -- which body and edition, which material
    category, which performance level -- and picking the wrong material category is
    an easy and consequential mistake: the transparency tables specify no color
    accuracy metric at all, while the reflective ones do.

    Grouping, level naming and the flags come from
    :mod:`deltachrome.profiles.catalogue`, so this listing and the desktop
    frontend's picker cannot describe the same profile differently.
    """
    from deltachrome.profiles.catalogue import catalogue, grouped
    from deltachrome.profiles.schema import load_profile

    def matches(summary) -> bool:
        if args.body and (summary.body or "").lower() != args.body.lower():
            return False
        if args.star is not None and summary.star_level != args.star:
            return False
        if args.material and args.material.lower() not in (summary.material or "").lower():
            return False
        if args.edition and args.edition.lower() not in (summary.edition or "").lower():
            return False
        return True

    selected = [s.profile_id for s in catalogue() if matches(s)]
    if not selected:
        print("No profile matches those filters. Available material categories:")
        for (body, edition, material), _ in grouped():
            print("  " + " ".join(p for p in (body, edition) if p)
                  + (f" — {material}" if material else ""))
        return EXIT_ERROR

    for (body, edition, material), group in grouped(selected):
        # A local guideline has no edition and no material category -- it has its
        # own name, which is what ``grouped`` files it under. Printing the dash
        # unconditionally left it reading "LOCAL GUIDELINE — ".
        print(f"{body} — {edition}" if edition else body)
        if material:
            print(f"  {material}")
        for summary in group:
            suffix = f"   [{'; '.join(summary.flags)}]" if summary.flags else ""
            print(f"    {summary.level_label}  {summary.profile_id}{suffix}")
            if args.metrics:
                profile = load_profile(summary.profile_id)
                for metric in profile.metrics:
                    condition = metric.pass_condition or "measured, not judged"
                    print(f"        {metric.metric_id}: {condition}")
    return EXIT_OK


def _cmd_metrics(_args: argparse.Namespace) -> int:
    from deltachrome.metrics import available_metrics

    for metric_id in available_metrics():
        print(metric_id)
    return EXIT_OK


def _cmd_inspect_reference(args: argparse.Namespace) -> int:
    from deltachrome.reference import read_reference

    reference = read_reference(args.path)
    print(f"{args.path.name}: {len(reference)} samples, {len(reference.fields)} fields")
    print(f"  fields: {', '.join(reference.fields)}")
    for key, value in reference.provenance().items():
        print(f"  {key}: {value}")
    if reference.measurement_mode:
        print(f"  measurement mode: {reference.measurement_mode}")
    for warning in reference.warnings:
        print(f"  warning: {warning}")

    if args.target:
        from deltachrome.targets.registry import load_target

        target = load_target(args.target)
        present, missing = reference.coverage(target.patch_ids)
        print(f"\n  coverage of {target.target_id}: {len(present)}/{len(target.patch_ids)}")
        if missing:
            print(f"  missing: {', '.join(missing[:20])}"
                  + (" ..." if len(missing) > 20 else ""))
        extra = sorted(set(reference.samples) - set(target.patch_ids))
        if extra:
            print(f"  in file but not in target: {', '.join(extra[:20])}"
                  + (" ..." if len(extra) > 20 else ""))
    return EXIT_OK


def _cmd_inspect_icc(args: argparse.Namespace) -> int:
    from deltachrome.imaging.loader import load_image

    image = load_image(args.path)
    print(image.summary())
    print(image.profile.summary())
    return EXIT_OK


# Kept as thin aliases: the parsers are shared run-input handling, not command
# line concerns, and both front ends must reject a bad value with the same words.
_parse_extent = parse_extent
_parse_corners = parse_corners


def _resolve_reference(args: argparse.Namespace, image=None, target=None):
    """Adapter over :func:`deltachrome.reference.resolve.resolve_reference`.

    The policy lives in the engine so that the desktop frontend reaches the same
    decision from the same code. All this adds is the CLI's own indentation.
    """
    from deltachrome.reference.resolve import resolve_reference
    from deltachrome.targets.registry import load_target

    choice = resolve_reference(
        target=target if target is not None else load_target(args.target),
        image=image if image is not None else args.image,
        explicit_path=args.reference,
        library_dir=args.reference_library,
        verify_code=not args.no_verify_code,
    )
    notes = [f"  {note}" for note in choice.notes]
    notes += [f"  WARNING: {warning}" for warning in choice.warnings]
    return choice.path, notes, choice.warnings


def _cmd_references(args: argparse.Namespace) -> int:
    from deltachrome.reference.library import ReferenceLibrary

    from deltachrome.home import REFERENCE_FILES, default_folder

    library = ReferenceLibrary(args.library or default_folder(REFERENCE_FILES))
    if args.bind:
        if "=" not in args.bind:
            print("--bind expects CODE=FILENAME", file=sys.stderr)
            return EXIT_ERROR
        code, filename = args.bind.split("=", 1)
        path = library.bind(code.strip(), filename.strip())
        print(f"bound {code.strip()} -> {filename.strip()} in {path}")
        return EXIT_OK

    if not library.entries:
        print(f"No reference files found in {args.library}")
        return EXIT_OK
    print(f"{len(library.entries)} reference file(s) in {args.library}:")
    for entry in library.entries:
        print(f"  {entry.describe()}")
        print(f"      answers to: {', '.join(sorted(entry.identifiers))}")
    return EXIT_OK


def _cmd_identify(args: argparse.Namespace) -> int:
    from deltachrome.imaging.codes import scan_codes
    from deltachrome.imaging.loader import load_image
    from deltachrome.reference.library import ReferenceLibrary

    image = load_image(args.image)
    scan = scan_codes(image, tiled=not args.no_tiles)
    print(f"{args.image.name}: {scan.describe()}")
    for code in scan.codes:
        print(f"  {code.text}  [{code.symbology} via {code.decoder}, {code.source}]")
    if scan.decoders_unavailable:
        print(f"  note: {', '.join(scan.decoders_unavailable)} not installed; "
              "1D barcodes and Data Matrix need it.")

    if args.library:
        resolution = ReferenceLibrary(args.library).resolve(
            scan.texts, standard_2of5=scan.standard_2of5)
        print(f"\n  {resolution.reason}")
        if resolution.path:
            print(f"  -> {resolution.path}")
        elif resolution.candidates:
            print("  candidates:")
            for entry in resolution.candidates:
                print(f"    {entry.describe()}")
    return EXIT_OK if scan.codes else EXIT_INCOMPLETE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
