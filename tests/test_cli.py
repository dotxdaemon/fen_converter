# ABOUTME: Tests the command-line interface behaviors for converting images to FEN.
# ABOUTME: Ensures conversion commands run successfully in non-interactive mode.
from pathlib import Path

from typer.testing import CliRunner

from fen_converter import cli


def test_convert_runs_without_interaction() -> None:
    runner = CliRunner()
    image = Path("examples/diagram.png")

    result = runner.invoke(cli.app, ["convert", str(image), "--no-interactive"])

    assert result.exit_code == 0, result.output
    assert "/" in result.stdout
    assert result.stdout.strip()
