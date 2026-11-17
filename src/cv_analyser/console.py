import io
import sys


def use_utf8_output() -> None:
    """Print any character, even on a Windows console.

    The Windows console defaults to the cp1252 code page, which cannot print
    characters models often write, such as a non-breaking hyphen or Bangla, and
    Python then crashes mid-report. Switch the output to UTF-8 and replace
    whatever the console still cannot show.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")
