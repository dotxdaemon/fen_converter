from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .board import BoardExtractionError, detect_board, extract_square_images
from .fen import build_fen
from .labeler import SquareClassifier, label_squares

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def convert(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, help="Input chessboard screenshot"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional path to store the generated FEN"),
    save_board: Optional[Path] = typer.Option(
        None,
        "--save-board",
        help="Save the rectified chessboard image for inspection.",
    ),
    interactive: bool = typer.Option(True, "--interactive", "-i", is_flag=True, help="Run the interactive labeler to confirm each square."),
    no_interactive: bool = typer.Option(False, "--no-interactive", "-n", is_flag=True, help="Skip interactive labeling."),
) -> None:
    """Convert a chessboard screenshot to a FEN string."""

    use_interactive = interactive and not no_interactive

    try:
        detected = detect_board(image)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc))
    except BoardExtractionError as exc:
        raise typer.BadParameter(f"Could not detect a chessboard: {exc}")

    if save_board:
        save_board.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        cv2.imwrite(str(save_board), detected.image)
        console.print(f"[green]Saved rectified board to {save_board}")

    squares = extract_square_images(detected.image)
    classifier = SquareClassifier()
    labels = label_squares(squares, classifier=classifier, interactive=use_interactive)
    fen = build_fen(labels)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(fen + "\n", encoding="utf-8")
        console.print(f"[green]FEN written to {output}")
    else:
        console.print(f"[bold cyan]{fen}")


@app.command()
def suggest(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Preview classifier suggestions without interactive prompts."""

    detected = detect_board(image)
    squares = extract_square_images(detected.image)
    classifier = SquareClassifier()

    table = Table(title="Square suggestions", show_header=True, header_style="bold magenta")
    table.add_column("Square")
    table.add_column("Symbol")
    table.add_column("Confidence")
    table.add_column("Reason")

    for square in sorted(squares.values(), key=lambda s: s.square, reverse=True):
        suggestion = classifier.suggest(square)
        table.add_row(
            suggestion.square,
            suggestion.symbol,
            f"{suggestion.confidence:.2f}",
            suggestion.reason,
        )
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
