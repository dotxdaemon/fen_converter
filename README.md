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

3. (Optional) Generate piece templates from a labelled board image if you want
   to override the built-in defaults:

   ```bash
   python generate_templates.py board.png "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
   ```
   This will create a `piece_templates.npz` file containing reference samples for each piece. Place the generated file alongside
   `convert.py` to make the converter use your custom data. Replace the arguments with an image and FEN that match the piece style
   you want to recognise.

4. Run the converter on a board screenshot:

   ```bash
   python convert.py path/to/board.png
   ```

The resulting FEN string is printed to stdout.
