"""CLI entry point (typer).

W1 scope: ``--version`` option and a ``hello`` placeholder command only.
The full CLI (``retrieve "query"``, interactive mode, ``history`` /
``config`` subcommands) will be wired up at the end of Phase 1 — see PRD
§2.8.
"""

import sys

import typer

from . import __version__

# Windows consoles default to GBK; force UTF-8 so CJK/emoji output works.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

app = typer.Typer(
    name="retrieve",
    help="Retriever — intelligent search & download agent (Phase 1 in progress).",
    add_completion=False,
)

def version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is passed."""
    if value:
        typer.echo(f"retrieve {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed retriever version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Retriever — intelligent search & download agent."""


@app.command()
def hello(
    source: str = typer.Option("example", "--source", "-s", help="Data source (reserved)."),
) -> None:
    """Print a welcome message. Full CLI lands at the end of Phase 1."""
    typer.echo("👋 Welcome to Retriever — 智能检索与下载 Agent！")
    typer.echo("Phase 1 开发中：models / config / http / base adapter 已就绪。")
    typer.echo(f"预留参数 --source={source}")


if __name__ == "__main__":
    app()
