#!/bin/sh
# Sign, notarize and staple Deltachrome.app, then its disk image.
#
#     packaging/sign_macos.sh dist/Deltachrome.app dist/Deltachrome-<version>-macos-<arch>.dmg
#
# SIGN_ONLY=1 stops after signing, for checking a signed build before Apple sees it.
#
# Needs, in the keychain of whoever runs it:
#   * a "Developer ID Application" certificate (the identity is found by that
#     name, or given as SIGN_IDENTITY), and
#   * notarization credentials stored with `xcrun notarytool store-credentials`
#     under the profile NOTARY_PROFILE (default: deltachrome-notary).
# Nothing secret passes through this script: it names the identity and the
# profile, and the keychain does the rest.
#
# Every Mach-O file is signed innermost first with the hardened runtime and a
# secure timestamp, which notarization requires. `codesign --deep` is not used:
# it signs nested code with the outer bundle's options and Apple advise against it.
set -eu
APP=$1
DMG=$2
cd "$(dirname "$0")/.."
PROFILE=${NOTARY_PROFILE:-deltachrome-notary}
# On a build machine there is no keychain profile: the App Store Connect API key
# is given as a file and its two IDs, from the repository's secrets.
if [ -n "${NOTARY_KEY:-}" ]; then
  AUTH="--key $NOTARY_KEY --key-id $NOTARY_KEY_ID --issuer $NOTARY_ISSUER"
else
  AUTH="--keychain-profile $PROFILE"
fi
ENTITLEMENTS=packaging/entitlements.plist
IDENTITY=${SIGN_IDENTITY:-$(security find-identity -v -p codesigning |
  sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)}
[ -n "$IDENTITY" ] || { echo "No Developer ID Application identity in the keychain."; exit 1; }
echo "Signing as: $IDENTITY"

sign() { codesign --force --timestamp --options runtime --sign "$IDENTITY" "$@"; }

# An application already signed by this identity, and intact, is not signed
# again: a new signature is a new binary to Apple, and would throw away a
# notarization already under way. SIGN_FORCE=1 signs regardless.
already_signed() {
  codesign --verify --strict --deep "$APP" 2>/dev/null &&
  codesign -dv --verbose=2 "$APP" 2>&1 | grep -q "Authority=$IDENTITY" &&
  codesign -dv "$APP" 2>&1 | grep -q "flags=.*runtime"
}
if [ "${SIGN_FORCE:-}" != 1 ] && already_signed; then
  echo "Already signed and intact; not signing again."
else

# 1. Every Mach-O file that is not a framework's own binary, deepest first.
find "$APP/Contents" -type f ! -path '*.framework/*' -print0 |
  xargs -0 file | grep 'Mach-O' | cut -d: -f1 |
  awk '{ print gsub("/", "/"), $0 }' | sort -rn | cut -d' ' -f2- |
  while IFS= read -r f; do sign "$f" >/dev/null 2>&1 || { echo "failed: $f"; sign "$f"; }; done

# 2. Frameworks, as bundles, deepest first (Qt's sit inside PySide6's folder).
find "$APP/Contents" -type d -name '*.framework' |
  awk '{ print gsub("/", "/"), $0 }' | sort -rn | cut -d' ' -f2- |
  while IFS= read -r fw; do sign "$fw" >/dev/null 2>&1 || { echo "failed: $fw"; sign "$fw"; }; done

# 3. The application itself, with its entitlements.
sign --entitlements "$ENTITLEMENTS" "$APP"
codesign --verify --strict --deep "$APP"
echo "Signed and verified."
fi
[ "${SIGN_ONLY:-}" = 1 ] && exit 0

# Submit once, remember the submission, and ask after it rather than holding one
# connection open: a first submission from a new account can sit in review for a
# day, and the credentials are unreadable while the screen is locked. Running
# the script again resumes a submission of the same file instead of making a new
# one, so a long review or an interrupted wait costs nothing.
notarize() {
  file=$1
  key=$2    # what identifies the content: the app's code hash, or the image's
  record="build/notary-$(basename "$file").id"
  sum=$key
  if [ -f "$record" ] && [ "$(cut -d' ' -f2 "$record")" = "$sum" ]; then
    id=$(cut -d' ' -f1 "$record")
    echo "Resuming submission $id for $(basename "$file")"
  else
    id=$(xcrun notarytool submit "$file" $AUTH --output-format plist |
         plutil -extract id raw -o - -)
    echo "$id $sum" > "$record"
    echo "Submitted $(basename "$file") as $id"
  fi
  # NOTARY_TIMEOUT_MIN caps the wait where waiting is billed by the minute: the
  # submission stays with Apple, and a later run of the same build resumes it.
  waited=0
  while :; do
    if [ -n "${NOTARY_TIMEOUT_MIN:-}" ] && [ "$waited" -ge "$NOTARY_TIMEOUT_MIN" ]; then
      echo "Still in review at Apple after $waited minutes: submission $id."
      exit 3
    fi
    waited=$((waited + 1))
    status=$(xcrun notarytool info "$id" $AUTH --output-format plist 2>/dev/null |
             plutil -extract status raw -o - - 2>/dev/null || echo "Unreachable")
    case "$status" in
      Accepted) echo "Accepted: $id"; return 0 ;;
      Invalid|Rejected)
        xcrun notarytool log "$id" $AUTH; exit 1 ;;
      *) printf '%s %s\n' "$(date +%H:%M)" "$status"; sleep 60 ;;
    esac
  done
}

# 4. Notarize the application and staple its ticket, so it opens offline too.
if xcrun stapler validate -q "$APP" 2>/dev/null; then
  echo "Application already notarized and stapled."
else
  CDHASH=$(codesign -dv --verbose=4 "$APP" 2>&1 | sed -n 's/^CDHash=//p')
  rm -f build/Deltachrome-notarize.zip
  ditto -c -k --keepParent "$APP" build/Deltachrome-notarize.zip
  notarize build/Deltachrome-notarize.zip "$CDHASH"
  xcrun stapler staple "$APP"
fi

# 5. The disk image, rebuilt around the stapled application, then itself
#    signed, notarized and stapled.
rm -rf build/dmg && mkdir -p build/dmg
cp -R "$APP" build/dmg/
cp LICENSE build/dmg/LICENSE.txt
cp build/THIRD_PARTY_NOTICES.txt build/dmg/
ln -s /Applications build/dmg/Applications
rm -f "$DMG"
hdiutil create -quiet -volname Deltachrome -srcfolder build/dmg -fs HFS+ -format ULMO "$DMG"
codesign --force --timestamp --sign "$IDENTITY" "$DMG"
notarize "$DMG" "$(shasum -a 256 "$DMG" | cut -d' ' -f1)"
xcrun stapler staple "$DMG"

# What a tester's Mac will decide.
spctl --assess --type execute --verbose=2 "$APP"
spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
echo "Done: $DMG"
