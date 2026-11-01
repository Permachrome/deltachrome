# Packaging

Builds the desktop application as installers with PyInstaller. Releases are
built by **GitHub Actions** (`.github/workflows/release.yml`): run *Build
installers* by hand, or push a version tag such as `v0.9.0b1`. It produces a
disk image for Apple silicon and one for Intel, both for macOS 13 and later,
and a zip for Windows, as downloadable artifacts. Publishing them is separate.

## What the scripts do

- `build_macos.sh` — the macOS app and disk image, for the Mac's own
  architecture. Installs every library from wheels chosen for macOS 13, and
  **fails if any bundled binary needs a newer macOS**. Needs a python.org
  Python 3.13 (`PYTHON=`): Homebrew's is built for the Mac it was installed on.
- `build_opencv.sh` — OpenCV built without FFmpeg or any video backend. The
  official macOS wheel links an FFmpeg configured `--enable-gpl` (x264, x265),
  so shipping it would distribute GPLv3 code for a feature this application
  never uses. About 25 minutes on an M-series Mac; cached in CI.
- `build_windows.ps1` — the Windows folder and zip. The official OpenCV wheel
  is used; its FFmpeg is a separate plugin, which the spec leaves out and the
  script checks is absent.
- `sign_macos.sh` — signs every binary with the hardened runtime, notarizes and
  staples the app, then builds, signs, notarizes and staples the disk image.
  Resumable: a submission is recorded against what it covers, and running the
  script again resumes it rather than uploading again. `entitlements.plist` is
  empty on purpose — the signed app needs no exceptions.
- `fetch_colprof.py` — ArgyllCMS's `colprof`, which builds camera profiles,
  from Argyll's own release archive for the machine's platform, refused unless
  its SHA-256 matches the one recorded in the script (Argyll publishes none).
  Shipped unmodified at `argyll/bin` in the bundle, where the application looks
  first, so a fresh install needs nothing set. The spec refuses to build
  without it. **No build carrying colprof may be published before
  Deltachrome's source is public** — the condition of Graeme Gill's written terms for bundling it.
- `export_public.py` — writes the public repository's tree from one commit of
  this one, over a clone of `Permachrome/deltachrome`: leaves out the private
  notes, `.github/` and the guide's working copy in `docs/site/`, ships this README as `docs/developer.md`
  behind the landing page in `public/`, and refuses if anything private or
  `data/` would be published. It prints the commit and push; it runs neither.
- `collect_notices.py` — `THIRD_PARTY_NOTICES.txt`, from the license files
  each bundled library installs, with the LGPL section and source offer for Qt,
  and a colprof section linking the ArgyllCMS website for its source.
- `deltachrome.spec` — the PyInstaller recipe. Several entries exist because
  the bundler cannot see an import or should not ship a file; each says which
  failure it prevents, and those failures are silent.
- `requirements-lock.txt` — the release's exact libraries, and why PySide6,
  NumPy and SciPy are the versions they are.
- `launch.py` — the entry point. `--cli` as the first argument runs the
  command-line tool inside the bundle, which is how a build is checked:

      dist/Deltachrome.app/Contents/MacOS/Deltachrome --cli analyze CAPTURE ...

  must write the same run record as `python -m deltachrome.cli analyze` from
  source. Do this on real captures before publishing; CI has none to use.

## Signing secrets

The workflow signs and notarizes when the repository has these secrets
(*Settings ▸ Secrets and variables ▸ Actions*), and builds unsigned otherwise:

| Secret | Contents |
|---|---|
| `MACOS_CERTIFICATE_P12` | the Developer ID Application certificate and key, exported from Keychain Access as `.p12`, then `base64 -i cert.p12 \| pbcopy` |
| `MACOS_CERTIFICATE_PASSWORD` | the password set on that export |
| `NOTARY_KEY_P8` | the text of the App Store Connect API key, `AuthKey_<id>.p8` |
| `NOTARY_KEY_ID` | that key's ID |
| `NOTARY_ISSUER_ID` | the team's issuer ID, from the same App Store Connect page |

Locally, signing uses the keychain instead: the certificate as installed by
Xcode, and notary credentials stored once with
`xcrun notarytool store-credentials deltachrome-notary`.

## Not done

- Windows code signing: an unsigned installer gets a SmartScreen warning.
- A Windows installer; the beta ships a zip.
- Trimming the bundle: Qt Quick, Qt PDF and the virtual keyboard are pulled in
  by Qt's image-format plugins and are not used.
