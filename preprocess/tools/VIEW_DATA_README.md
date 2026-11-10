# view_data.py — Parquet/Arrow File Viewer

A terminal UI utility for viewing Parquet and Arrow files in a read-only mode, similar to `nano`. Features a frozen header row that stays visible while scrolling through data.

## Usage

```bash
python -m preprocess.tools.view_data <path_to_file.parquet|.arrow> [-n ROWS] [--load-limit N]
```

### Options

- `<file>` — Path to a Parquet (`.parquet`) or Arrow (`.arrow`) file
- `-n, --rows` — Rows loaded per chunk; more are loaded on demand as you scroll (default: 100)
- `--load-limit` — Prompt before loading beyond this many rows, since holding the whole file in memory can be slow (default: 1000)

### Incremental loading

Only the first chunk of rows is loaded up front, so even multi-GB files open instantly. As you scroll past the loaded rows, the next chunk is pulled in automatically. Once you cross `--load-limit` rows, the viewer asks for confirmation (`y`/`n`) before loading further — declining keeps you at the rows already loaded. The title bar shows `loaded / total` (a trailing `+` means more remain).

### Examples

```bash
# View a parquet file (loads 100 rows at a time)
python -m preprocess.tools.view_data data.parquet

# Load 500 rows per chunk, and ask before going beyond 5000
python -m preprocess.tools.view_data data.arrow -n 500 --load-limit 5000
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `Ctrl+C` | Quit the viewer |
| `↑` / `↓` | Scroll up/down by 1 row |
| `Page Up` / `Page Down` | Scroll by full screen height |
| `Home` / `End` | Jump to the first / last loaded row |
| `←` / `→` | Select the previous / next column (clamps at the edges) |
| `Esc` | Deselect the focused column |
| `w` | Toggle line wrap on the focused column |
| `t` | Toggle thousand separators |
| `d` | Toggle the data-type row |
| `c` | Toggle colors for special diploid tokens (seq column) |

## Features

- **Frozen header** — Column names stay visible at the top while scrolling (like Excel's freeze panes)
- **Automatic column sizing** — Columns fit the terminal width; long values are truncated with `…`
- **Read-only** — Safe to view any file without risk of modification
- **Supports multiple formats** — Reads both Parquet and Arrow IPC files
- **Status bar** — Shows current row position and column count

## Display Example

```
 height_gwas_clean.parquet (100 rows, 11 cols)                              
├───────────┼───────────┼───────────┼───────────┼───────────┼──────────────
│ SNP       │ RSID      │ CHR       │ BP        │ A1        │ A2
├───────────┼───────────┼───────────┼───────────┼───────────┼──────────────
│ 20:341805 │ rs1886695 │ 20        │ 34180535  │ A         │ G
│ 20:342265 │ rs6025   │ 20        │ 34226554  │ C         │ T
│ 20:342278 │ rs2019266 │ 20        │ 34227803  │ G         │ A
...
 Row 1/100 | Press 'q' to quit, ↑/↓ or PgUp/PgDn to scroll
```

## Implementation Details

- Uses Python's built-in `curses` library for terminal UI
- Reads files with PyArrow (efficient columnar access)
- Thread-safe initialization (file reading before UI starts)
- Non-blocking keyboard input with 100ms refresh rate

## Testing

```bash
uv run --no-sync pytest preprocess/tests/test_view_data.py -v
```

All tests pass including:
- Parquet and Arrow file loading
- Row limiting
- Cell formatting and truncation
- Header and separator rendering
- Error handling for missing/unsupported files
