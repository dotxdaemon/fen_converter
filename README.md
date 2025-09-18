# Chess FEN Converter

This utility converts a cropped chessboard image into FEN notation. It expects the image to contain only the board (8x8 squares) with standard piece designs. The conversion uses computer vision to detect piece shapes.

## Usage

1. Install the Cairo graphics library (required by `cairosvg`):
   - macOS: `brew install cairo`
   - Debian/Ubuntu: `sudo apt-get install libcairo2`

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Generate piece templates from a labelled board image if you want to override the built-in defaults:

   ```bash
   python generate_templates.py board.png "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
   ```
   This will create a `piece_templates.npz` file containing reference samples for each piece. Place the generated file alongside `convert.py` to make the converter use your custom data. Replace the arguments with an image and FEN that match the piece style you want to recognise.

4. Run the converter on a board screenshot:

   ```bash
   python convert.py path/to/board.png
   ```

   The resulting FEN string is printed to stdout.

## Using the enhanced converter on full screenshots

In situations where you have a full window capture with borders, arrows or flipped boards (for example, Chess.com game screenshots), the original converter will not work out of the box.  To handle these cases, this repository includes an alternate script `fen_fix.py` and a set of prebuilt templates for the Chess.com piece style (`piece_templates_chesscom.npz`).

The enhanced converter automatically crops the chessboard from the screenshot, tests all four orientations and uses the supplied template archive to detect pieces.  To run it on a Chess.com screenshot use:

```bash
python fen_fix.py --image path/to/screenshot.png --templates piece_templates_chesscom.npz
```

If your screenshot uses a different piece design, generate templates from a labelled board image as described above and provide the resulting `.npz` file via the `--templates` option.

Like `convert.py`, `fen_fix.py` will print the resulting FEN string to stdout.
