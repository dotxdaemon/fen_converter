# Screenshot to FEN Converter

This small utility converts a cropped chessboard image into FEN notation. It expects the image to contain only the board (8x8 squares) with standard piece designs. The conversion uses template matching with piece images generated from `python-chess`.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the converter on a board screenshot:
   ```bash
   python fen_converter/convert.py path/to/board.png
   ```
   The resulting FEN string is printed to stdout.

The approach is naive and works best with clear board images and the default piece style.
