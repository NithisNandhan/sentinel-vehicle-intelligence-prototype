"""Write the permission receipt that lets a downloaded video be analysed.

Run this only for footage you have explicit permission to download and use.
The receipt records where the file came from and its SHA256; it never
overwrites an existing receipt. Example:

  python tools/register_recording.py data/recordings/junction.mp4 \
      --source "https://example.org/dataset" --licence "CC BY 4.0" \
      --permission "Dataset licence permits download and research use"
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))

from sentinel_edge.recorded_source import RecordingError, write_receipt  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('video', type=Path)
    parser.add_argument('--source', required=True, help='Where the file was obtained (URL or description)')
    parser.add_argument('--licence', required=True)
    parser.add_argument('--permission', required=True, help='Why you may download and use it')
    parser.add_argument('--title')
    parser.add_argument('--obtained-on', help='Date obtained, YYYY-MM-DD')
    parser.add_argument('--recorded-start-utc',
                        help='ISO 8601 UTC time of the first frame, only if genuinely known')
    args = parser.parse_args()
    try:
        receipt = write_receipt(args.video, source=args.source, licence=args.licence,
                                permission=args.permission, title=args.title,
                                obtained_on=args.obtained_on, recorded_start_utc=args.recorded_start_utc)
    except FileExistsError:
        raise SystemExit('A receipt already exists for this file; it was not changed.')
    except RecordingError as error:
        raise SystemExit(f'Not registered: {error}')
    print(f'Receipt written: {receipt.name}')
