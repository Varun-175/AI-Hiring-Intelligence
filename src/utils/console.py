import os
import sys


class ConsoleStyle:
    RESET = "\033[0m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"


def _ensure_utf8_stdout():
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


_ensure_utf8_stdout()


def _supports_color():
    return os.getenv("NO_COLOR") is None


def color(text, tone="white", bold=False):
    if not _supports_color():
        return text

    tone_code = {
        "cyan": ConsoleStyle.CYAN,
        "blue": ConsoleStyle.BLUE,
        "green": ConsoleStyle.GREEN,
        "yellow": ConsoleStyle.YELLOW,
        "red": ConsoleStyle.RED,
        "white": ConsoleStyle.WHITE,
    }.get(tone, ConsoleStyle.WHITE)

    weight = ConsoleStyle.BOLD if bold else ""
    return f"{weight}{tone_code}{text}{ConsoleStyle.RESET}"


def separator(char="=", width=80, tone="cyan"):
    print(color(char * width, tone=tone))


def header(title, width=80):
    separator("=", width=width, tone="cyan")
    print(color(title.center(width), tone="cyan", bold=True))
    separator("=", width=width, tone="cyan")


def stage(title):
    print()
    print(color(title, tone="blue", bold=True))


def status(label, value=None, tone="green", symbol="✓", indent=4):
    prefix = " " * indent
    safe_symbol = symbol
    if (getattr(sys.stdout, "encoding", "") or "").lower() not in {"utf-8", "utf8"}:
        safe_symbol = "*" if symbol not in {"✗"} else "x"
    if value is None:
        print(f"{prefix}{color(safe_symbol, tone=tone, bold=True)} {color(label, tone='white')}")
        return

    print(
        f"{prefix}{color(safe_symbol, tone=tone, bold=True)} "
        f"{color(f'{label:<20}', tone='white')} : {color(str(value), tone='white')}"
    )


def detail(label, value, indent=4):
    prefix = " " * indent
    print(
        f"{prefix}{color(f'{label:<20}', tone='white')} : "
        f"{color(str(value), tone='white')}"
    )


def message(text, tone="white", indent=4, bullet=None):
    prefix = " " * indent
    if bullet:
        print(f"{prefix}{color(bullet, tone=tone, bold=True)} {color(text, tone=tone)}")
    else:
        print(f"{prefix}{color(text, tone=tone)}")


def section_break(width=80):
    print()
    print(color("-" * width, tone="cyan"))
