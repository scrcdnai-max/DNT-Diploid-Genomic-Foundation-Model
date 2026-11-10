#!/usr/bin/env python3
"""
Utility for viewing Parquet and Arrow files in a terminal UI (similar to nano).
Loads the first chunk of rows instantly; more rows are pulled in on demand as
you scroll. Past --load-limit rows (default 1000) it asks before loading more,
since holding the whole file in memory can be slow.

Usage:
    python -m preprocess.tools.view_data <path_to_file.parquet|.arrow> [-n ROWS] [--load-limit N]

Keys:
    q          Quit
    ↑ / ↓      Scroll rows (loads more rows when scrolling past the loaded ones)
    PgUp/PgDn  Scroll by page
    Home/End   Jump to first / last loaded row
    ← / →      Select previous/next column
    Esc        Deselect column
    w          Toggle line wrap on selected column
    t          Toggle thousand separators
    d          Toggle data-type row
    c          Toggle colors for special diploid tokens (seq column)
"""

import curses
import sys
import argparse
from pathlib import Path
from datetime import datetime

import pyarrow.parquet as pq
import pyarrow as pa

INT_TYPES = frozenset({
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
})

# Special diploid tokens, grouped by role (source: diploid_tokens.tsv).
# Plain reference bases A/C/G/T are deliberately omitted — they render in the
# normal data colour so the eye is drawn to the variant tokens.
_TOK_UNPHASED   = frozenset('ÅĄĈČĜĢŢȺȻȾ')           # heterozygous, phase unknown
_TOK_PHASED     = frozenset('ÁÃÆÇĂĆĊĞŤƇƓƬǤǦȚДКТШг')  # heterozygous, phased
_TOK_MARKERS    = frozenset('^{}')                   # SNP / INDEL span boundaries
_TOK_DIRECTION  = frozenset('‹›')                    # phase-direction indicators
_TOK_DELETION   = frozenset('-*')                    # deletion / spanning deletion
_TOK_HEMIZYGOUS = frozenset('.acgnt·')               # single allele (chrX/Y)
_TOK_AMBIGUOUS  = frozenset('N')                     # ambiguous base

# Role → ordering used to assign colour pairs; also the keys of the colour maps.
_TOKEN_ROLE_ORDER = ('unphased', 'phased', 'marker', 'direction',
                     'deletion', 'hemizygous', 'ambiguous')
_TOKEN_SETS = {
    'unphased': _TOK_UNPHASED, 'phased': _TOK_PHASED, 'marker': _TOK_MARKERS,
    'direction': _TOK_DIRECTION, 'deletion': _TOK_DELETION,
    'hemizygous': _TOK_HEMIZYGOUS, 'ambiguous': _TOK_AMBIGUOUS,
}
# Fixed xterm-256 indices (consistent across terminals) and a 16-colour fallback.
_ROLE_COLOR_256 = {'unphased': 214, 'phased': 114, 'marker': 175, 'direction': 67,
                   'deletion': 203, 'hemizygous': 80, 'ambiguous': 244}

# Columns whose cells are diploid sequences and get per-token colouring.
SEQUENCE_COLUMN_NAMES = frozenset({'seq'})

# Stop measuring content width past this — no terminal is wider, so the layout
# treats anything beyond it as "very wide" regardless of the exact length.
_MAX_MEASURE = 1024


def token_role(ch: str):
    """Return the diploid-token role of a character, or None for plain
    reference bases and any other (non-token) character."""
    for role in _TOKEN_ROLE_ORDER:
        if ch in _TOKEN_SETS[role]:
            return role
    return None


class DataViewer:
    def __init__(self, file_path: str, max_rows: int = 100, load_limit: int = 1000):
        self.file_path = Path(file_path)
        self.max_rows = max_rows        # rows revealed per chunk (initial + each load)
        self.chunk = max_rows
        self.load_limit = load_limit    # ask before loading beyond this many rows
        self.scroll_offset = 0
        self.focused_col = None     # int or None
        self.wrap_focused = False
        self.show_thousands = True
        self.show_dtypes = True
        self.show_token_colors = True

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._load_data()

    # ------------------------------------------------------------------
    # Data loading (incremental — rows are pulled in as they are scrolled into)
    # ------------------------------------------------------------------

    def _load_data(self):
        """Open the file and reveal the first chunk. Further rows are pulled in
        on demand by _load_more() as the user scrolls past the loaded rows."""
        suffix = self.file_path.suffix.lower()
        if suffix == '.parquet':
            self._pf = pq.ParquetFile(self.file_path)
            self.total_rows = self._pf.metadata.num_rows
            self._schema = self._pf.schema_arrow
            self._batch_gen = self._pf.iter_batches(batch_size=self.chunk)
        elif suffix == '.arrow':
            self._fh = open(self.file_path, 'rb')
            reader = pa.ipc.open_stream(self._fh)
            self._schema = reader.schema
            self.total_rows = None       # unknown until the stream is exhausted
            self._batch_gen = iter(reader)
        else:
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}")

        self.table = self._schema.empty_table()
        self._buffer = self._schema.empty_table()   # pulled-but-not-yet-revealed
        self._fully_loaded = False
        self._load_paused = False                    # user declined to load more
        self._load_confirmed = False                 # user OK'd loading past the limit

        # Schema-derived metadata is fixed; only row-count state changes on load.
        self.num_cols = len(self._schema.names)
        self.columns = list(self._schema.names)
        self.col_dtypes = [str(f.type) for f in self._schema]
        self._seq_columns = frozenset(
            i for i, name in enumerate(self.columns)
            if name in SEQUENCE_COLUMN_NAMES
        )
        self.num_rows = 0
        self.linenum_width = 1
        self._content_width_cache = {}

        self._load_more()  # first chunk

    def close(self):
        """Close the underlying file handle (arrow stream is kept open for
        on-demand reads). Safe to call more than once."""
        fh = getattr(self, '_fh', None)
        if fh is not None:
            fh.close()
            self._fh = None

    def _refill_buffer(self):
        """Pull the next non-empty source batch into the buffer, or mark EOF."""
        for batch in self._batch_gen:
            if batch.num_rows:
                self._buffer = pa.Table.from_batches([batch], schema=self._schema)
                return
        self._fully_loaded = True

    def _reveal(self, n: int) -> int:
        """Move up to n more rows from the source into self.table. Returns the
        number of rows actually added (0 at end of file)."""
        parts = [self.table] if self.table.num_rows else []
        moved = 0
        while moved < n:
            if self._buffer.num_rows == 0:
                if self._fully_loaded:
                    break
                self._refill_buffer()
                if self._buffer.num_rows == 0:
                    break
            take = min(n - moved, self._buffer.num_rows)
            parts.append(self._buffer.slice(0, take))
            self._buffer = self._buffer.slice(take)
            moved += take
        if moved:
            self.table = pa.concat_tables(parts)
        return moved

    def _load_more(self) -> int:
        """Reveal one more chunk of rows; refresh row-count-dependent state."""
        added = self._reveal(self.chunk)
        self.num_rows = self.table.num_rows
        self.linenum_width = len(str(self.total_rows or self.num_rows or 1))
        # When the total is known (parquet), reaching it means we are done —
        # no need to probe for one more batch to discover end-of-file.
        if self.total_rows is not None and self.num_rows >= self.total_rows:
            self._fully_loaded = True
        if added:
            self._content_width_cache = {}   # new rows may widen a column
        return added

    def _ensure_loaded(self, up_to_row: int, confirm) -> None:
        """Load chunks until up_to_row rows are available. Past load_limit,
        `confirm()` is called once; if it returns False, loading is paused."""
        while not self._fully_loaded and self.num_rows < up_to_row:
            if (not self._load_confirmed) and self.num_rows >= self.load_limit:
                if not confirm():
                    self._load_paused = True
                    return
                self._load_confirmed = True
            self._load_more()

    # ------------------------------------------------------------------
    # Cell formatting
    # ------------------------------------------------------------------

    def _raw_cell_content(self, value, dtype: str = "") -> str:
        """Full string representation of a value (no truncation)."""
        if value is None:
            return "None"
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        if isinstance(value, (list, dict)):
            return str(value)
        if getattr(self, 'show_thousands', False) and dtype in INT_TYPES and isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def _format_cell(self, value, width: int, dtype: str = "") -> str:
        """Format a cell value to exactly `width` chars, truncating with … if needed."""
        s = self._raw_cell_content(value, dtype)
        if len(s) > width:
            s = s[:width - 1] + "…"
        return s.ljust(width)

    def _abbrev_dtype(self, dtype_str: str) -> str:
        """Short display label for an Arrow type string."""
        exact = {
            'int8': 'int8',   'int16': 'int16',  'int32': 'int32',  'int64': 'int64',
            'uint8': 'uint8', 'uint16': 'uint16', 'uint32': 'uint32', 'uint64': 'uint64',
            'float16': 'f16', 'float32': 'f32',  'float64': 'f64',  'double': 'f64',
            'string': 'str',  'large_string': 'str', 'utf8': 'str', 'large_utf8': 'str',
            'bool': 'bool',   'boolean': 'bool',
            'binary': 'bytes', 'large_binary': 'bytes',
            'null': 'null',   'date32[day]': 'date', 'date64[ms]': 'date',
        }
        if dtype_str in exact:
            return exact[dtype_str]
        if dtype_str.startswith('timestamp'):
            return 'ts'
        if dtype_str.startswith('list<'):
            return 'list'
        if dtype_str.startswith('struct<'):
            return 'struct'
        if dtype_str.startswith('map<'):
            return 'map'
        if dtype_str.startswith('dict'):
            return 'dict'
        return dtype_str[:8]

    # ------------------------------------------------------------------
    # Column widths
    # ------------------------------------------------------------------

    def _content_width(self, col_idx: int) -> int:
        """Width needed to show this column without truncation: the longest
        rendered cell across the loaded rows, and the header name.

        Cached per (column, thousand-separator state); invalidated when more
        rows are revealed. Measurement stops at _MAX_MEASURE — beyond that the
        column is "very wide" and the layout caps it anyway, so there is no need
        to materialise a list column's full thousands-of-chars string per cell.
        """
        if not (0 <= col_idx < self.num_cols):
            return 0
        cache = self.__dict__.setdefault('_content_width_cache', {})
        key = (col_idx, getattr(self, 'show_thousands', False))
        if key in cache:
            return cache[key]
        dtype = self.col_dtypes[col_idx] if col_idx < len(self.col_dtypes) else ""
        # Fit everything shown in the column: header, dtype label, and data.
        width = max(len(self.columns[col_idx]), len(self._abbrev_dtype(dtype)))
        for scalar in self.table.column(col_idx):
            value = scalar.as_py() if scalar.is_valid else None
            width = max(width, len(self._raw_cell_content(value, dtype)))
            if width >= _MAX_MEASURE:
                width = _MAX_MEASURE
                break
        cache[key] = width
        return width

    def _get_column_widths(self, max_total_width: int) -> list:
        """
        Distribute column widths across the terminal so the row fills the
        screen without overflowing.

        Each narrow column (content < a fair share of the width) shrinks to its
        content; the slack they free is split evenly among the wide columns
        (content > fair share), each capped at its own content. With a very wide
        column present (e.g. seq), this fills the screen by default — the same
        layout you get by focusing that column. The focused column, if still
        truncated, takes priority: the others shrink and it absorbs the rest.
        """
        if self.num_cols == 0:
            return []
        lw = getattr(self, 'linenum_width', 0)
        # Row structure overhead: │ NNN │  + (num_cols × 3 for separators)
        fixed = (lw + 4 if lw > 0 else 0) + self.num_cols * 3
        available = max(self.num_cols * 4, max_total_width - fixed)
        fair_share = max(4, available // self.num_cols)

        autosize = [self._content_width(i) for i in range(self.num_cols)]

        # Narrow columns take their content; wide columns start at a fair share
        # and absorb the slack the narrow columns leave behind.
        widths = [max(4, min(a, fair_share)) for a in autosize]
        wide = [i for i, a in enumerate(autosize) if a > fair_share]
        leftover = available - sum(widths)
        if wide and leftover > 0:
            per, rem = divmod(leftover, len(wide))
            for k, i in enumerate(wide):
                widths[i] = min(autosize[i], widths[i] + per + (1 if k < rem else 0))

        # Focused column: if still truncated, give it the screen — the others
        # shrink to content (wide ones to a fair share) and it takes the rest.
        fc = getattr(self, 'focused_col', None)
        if fc is not None and 0 <= fc < self.num_cols and autosize[fc] > widths[fc]:
            for i in range(self.num_cols):
                if i != fc:
                    widths[i] = max(4, min(autosize[i], fair_share))
            other_sum = sum(w for i, w in enumerate(widths) if i != fc)
            widths[fc] = max(widths[fc], min(autosize[fc], available - other_sum))

        return widths

    # ------------------------------------------------------------------
    # String-based renderers (used by tests; not called during display)
    # ------------------------------------------------------------------

    def _render_separator(self, col_widths: list) -> str:
        seps = ["─" * (w + 2) for w in col_widths]
        lw = getattr(self, 'linenum_width', 0)
        if lw > 0:
            return "├" + "─" * (lw + 2) + "┼" + "┼".join(seps) + "┤"
        return "├" + "┼".join(seps) + "┤"

    def _render_header(self, col_widths: list) -> str:
        cells = [self._format_cell(name, w) for name, w in zip(self.columns, col_widths)]
        lw = getattr(self, 'linenum_width', 0)
        prefix = f"│ {' ' * lw} │ " if lw > 0 else "│ "
        return prefix + " │ ".join(cells) + " │"

    def _render_dtype_row(self, col_widths: list) -> str:
        dtypes = getattr(self, 'col_dtypes', [''] * self.num_cols)
        cells = [self._format_cell(self._abbrev_dtype(dt), w)
                 for dt, w in zip(dtypes, col_widths)]
        lw = getattr(self, 'linenum_width', 0)
        prefix = f"│ {' ' * lw} │ " if lw > 0 else "│ "
        return prefix + " │ ".join(cells) + " │"

    def _render_data_row(self, row_idx: int, col_widths: list) -> str:
        if row_idx >= self.num_rows:
            return ""
        row = self.table.slice(row_idx, 1)
        dtypes = getattr(self, 'col_dtypes', [''] * self.num_cols)
        cells = [self._format_cell(row[i][0].as_py() if row[i][0].is_valid else None,
                                   w, dtypes[i] if i < len(dtypes) else "")
                 for i, w in enumerate(col_widths)]
        lw = getattr(self, 'linenum_width', 0)
        if lw > 0:
            return f"│ {row_idx + 1:>{lw}} │ " + " │ ".join(cells) + " │"
        return "│ " + " │ ".join(cells) + " │"

    # ------------------------------------------------------------------
    # Screen-writing helpers
    # ------------------------------------------------------------------

    def _write_row(self, stdscr, srow: int, col_widths: list, max_w: int,
                   default_attr: int, sep_attr: int,
                   linenum_str: str, cell_provider,
                   color_thousand_seps: bool = False,
                   color_tokens: bool = False) -> None:
        """
        Write one screen line.
        cell_provider(col_i, width) -> (text, attr).
        When color_thousand_seps=True, commas in integer cells are rendered in sep_attr.
        When color_tokens=True, special diploid tokens in sequence columns are
        coloured per-character by their role.
        """
        x = 0
        lw = self.linenum_width

        def put(text: str, attr: int = default_attr) -> None:
            nonlocal x
            if x >= max_w or not text:
                return
            s = text[:max_w - x]
            stdscr.addstr(srow, x, s, attr)
            x += len(s)

        put(f"│ {linenum_str} │ ")

        for col_i, width in enumerate(col_widths):
            if col_i > 0:
                put(" │ ")
            text, attr = cell_provider(col_i, width)
            dt = self.col_dtypes[col_i] if col_i < len(self.col_dtypes) else ""
            if (color_tokens and
                    getattr(self, 'show_token_colors', False) and
                    col_i in getattr(self, '_seq_columns', ())):
                prev = ''
                for ch in text:
                    # A plain base right after a ^ SNP marker is itself the
                    # (homozygous) SNP allele — colour it like a diploid token
                    # (the yellow-orange gold) so homozygous SNPs stand out too.
                    if prev == '^' and token_role(ch) is None and ch.isalpha():
                        put(ch, self._role_attr_for('unphased', attr))
                    else:
                        put(ch, self._token_attr(ch, attr))
                    prev = ch
            elif (color_thousand_seps and
                    getattr(self, 'show_thousands', False) and
                    dt in INT_TYPES and ',' in text):
                for ch in text:
                    put(ch, sep_attr if ch == ',' else attr)
            else:
                put(text, attr)

        put(" │")

    def _role_attr_for(self, role, base_attr: int) -> int:
        """Attr for a token role (None role → base attr unchanged), keeping the
        base attr's bold bit."""
        attr = getattr(self, '_role_attrs', {}).get(role)
        if attr is None:
            return base_attr
        return attr | (base_attr & curses.A_BOLD)

    def _token_attr(self, ch: str, base_attr: int) -> int:
        """Colour for a special diploid token; plain reference bases and
        non-token characters keep the base attr."""
        return self._role_attr_for(token_role(ch), base_attr)

    def _write_header_row(self, stdscr, srow: int, col_widths: list, max_w: int,
                          header_attr: int, focus_attr: int, sep_attr: int) -> None:
        fc = getattr(self, 'focused_col', None)
        lw = self.linenum_width

        def cell_provider(col_i, width):
            text = self._format_cell(self.columns[col_i], width)
            return text, (focus_attr if col_i == fc else header_attr)

        self._write_row(stdscr, srow, col_widths, max_w,
                        header_attr, sep_attr, ' ' * lw, cell_provider)

    def _write_dtype_row_screen(self, stdscr, srow: int, col_widths: list, max_w: int,
                                dtype_attr: int, focus_attr: int, sep_attr: int) -> None:
        fc = getattr(self, 'focused_col', None)
        lw = self.linenum_width

        def cell_provider(col_i, width):
            text = self._format_cell(self._abbrev_dtype(self.col_dtypes[col_i]), width)
            return text, (focus_attr if col_i == fc else dtype_attr)

        self._write_row(stdscr, srow, col_widths, max_w,
                        dtype_attr, sep_attr, ' ' * lw, cell_provider)

    def _write_data_row(self, stdscr, start_srow: int, row_idx: int,
                        col_widths: list, max_w: int,
                        row_attr: int, sep_attr: int, focus_attr: int,
                        max_screen_rows: int = 1) -> int:
        """
        Write a data row. Returns number of screen lines consumed.
        When wrap_focused=True and a column is focused, the focused column's
        full content wraps across as many lines as needed (capped at
        max_screen_rows).
        """
        if row_idx >= self.num_rows:
            return 1

        table_row = self.table.slice(row_idx, 1)
        fc = getattr(self, 'focused_col', None)
        wrap = getattr(self, 'wrap_focused', False) and fc is not None

        # Pre-compute wrap chunks for the focused column
        if wrap and fc is not None and fc < self.num_cols:
            cell_val = table_row[fc][0].as_py() if table_row[fc][0].is_valid else None
            raw = self._raw_cell_content(cell_val, self.col_dtypes[fc])
            fw = max(1, col_widths[fc])
            chunks = [raw[i:i + fw].ljust(fw) for i in range(0, max(1, len(raw)), fw)]
            num_lines = min(len(chunks), max_screen_rows)
        else:
            chunks = None
            num_lines = 1

        lw = self.linenum_width

        for line in range(num_lines):
            srow = start_srow + line
            is_first = (line == 0)
            linenum_str = f"{row_idx + 1:>{lw}}" if is_first else ' ' * lw

            def make_provider(line_=line, is_first_=is_first):
                def cell_provider(col_i, width):
                    if col_i == fc:
                        if wrap and chunks is not None:
                            chunk = chunks[line_] if line_ < len(chunks) else ' ' * width
                            return chunk, focus_attr
                        else:
                            if is_first_:
                                val = table_row[col_i][0].as_py() if table_row[col_i][0].is_valid else None
                                return self._format_cell(val, width, self.col_dtypes[col_i]), focus_attr | curses.A_BOLD
                            return ' ' * width, row_attr
                    else:
                        if is_first_:
                            val = table_row[col_i][0].as_py() if table_row[col_i][0].is_valid else None
                            dt = self.col_dtypes[col_i] if col_i < len(self.col_dtypes) else ""
                            return self._format_cell(val, width, dt), row_attr
                        return ' ' * width, row_attr
                return cell_provider

            self._write_row(stdscr, srow, col_widths, max_w,
                            row_attr, sep_attr, linenum_str,
                            make_provider(), color_thousand_seps=True,
                            color_tokens=True)

        return num_lines

    # ------------------------------------------------------------------
    # Column focus navigation
    # ------------------------------------------------------------------

    def _focus_left(self) -> None:
        """Move focus one column left; clamp at the left-most column (no wrap).

        With no column focused, pressing left enters from the right edge.
        """
        if self.num_cols == 0:
            return
        if self.focused_col is None:
            self.focused_col = self.num_cols - 1
        else:
            self.focused_col = max(0, self.focused_col - 1)

    def _focus_right(self) -> None:
        """Move focus one column right; clamp at the right-most column (no wrap).

        With no column focused, pressing right enters from the left edge.
        """
        if self.num_cols == 0:
            return
        if self.focused_col is None:
            self.focused_col = 0
        else:
            self.focused_col = min(self.num_cols - 1, self.focused_col + 1)

    def _confirm(self, stdscr, message: str) -> bool:
        """Show a y/n prompt on the last line and block until the user answers.
        Returns True for yes, False for no / Esc / q."""
        max_y, max_x = stdscr.getmaxyx()
        try:
            stdscr.addstr(max_y - 1, 0, message[:max_x - 1].ljust(max_x - 1),
                          curses.A_REVERSE | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.refresh()
        while True:
            ch = stdscr.getch()
            if ch in (ord('y'), ord('Y')):
                return True
            if ch in (ord('n'), ord('N'), 27, ord('q')):
                return False

    # ------------------------------------------------------------------
    # Main display loop
    # ------------------------------------------------------------------

    def display(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)

        # ── Cohesive "slate + amber" palette ──────────────────────────────
        # Header: crisp white on deep slate-blue (high contrast, calm).
        # Data + borders: light gray on the terminal's own background.
        # Secondary info (dtype row, thousand separators): muted sage/gray.
        # Accent (focused column): ink on amber — the single highlight color.
        # Falls back to the basic 8-color set on terminals without 256 colors.
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK

        # Diploid token roles get fixed xterm-256 indices (consistent across
        # terminals); a 16-colour fallback keeps the feature usable elsewhere.
        role_fallback = {'unphased': curses.COLOR_YELLOW, 'phased': curses.COLOR_GREEN,
                         'marker': curses.COLOR_MAGENTA, 'direction': curses.COLOR_CYAN,
                         'deletion': curses.COLOR_RED, 'hemizygous': curses.COLOR_CYAN,
                         'ambiguous': curses.COLOR_WHITE}

        if curses.COLORS >= 256:
            curses.init_pair(1, 231, 24)    # header  — white on slate-blue
            curses.init_pair(2, 252, bg)    # data    — light gray
            curses.init_pair(3, 222, bg)    # title / status — soft amber
            curses.init_pair(4, 240, bg)    # thousand separators — dim gray
            curses.init_pair(5, 108, bg)    # dtype row — muted sage
            curses.init_pair(6, 234, 179)   # focused header — ink on amber
            for k, role in enumerate(_TOKEN_ROLE_ORDER):
                curses.init_pair(7 + k, _ROLE_COLOR_256[role], bg)
        else:
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # column names
            curses.init_pair(2, curses.COLOR_WHITE, bg)                   # normal data
            curses.init_pair(3, curses.COLOR_YELLOW, bg)                  # title / status
            curses.init_pair(4, curses.COLOR_BLUE, bg)                    # thousand separators
            curses.init_pair(5, curses.COLOR_CYAN, bg)                    # dtype row
            curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # focused column header
            for k, role in enumerate(_TOKEN_ROLE_ORDER):
                curses.init_pair(7 + k, role_fallback[role], bg)

        col_attr        = curses.color_pair(1) | curses.A_BOLD
        data_attr       = curses.color_pair(2)
        stat_attr       = curses.color_pair(3)
        sep_attr        = curses.color_pair(4)
        dtype_attr      = curses.color_pair(5)
        focus_hdr_attr  = curses.color_pair(6) | curses.A_BOLD
        focus_data_attr = curses.color_pair(2) | curses.A_BOLD
        self._role_attrs = {role: curses.color_pair(7 + k)
                            for k, role in enumerate(_TOKEN_ROLE_ORDER)}

        while True:
            max_y, max_x = stdscr.getmaxyx()
            header_rows = 5 if self.show_dtypes else 4
            data_start = header_rows
            content_height = max(0, max_y - header_rows - 1)

            # Reveal rows for the viewport (and one row ahead) as it is scrolled
            # into; past load_limit, prompt before loading more.
            if not self._load_paused:
                self._ensure_loaded(
                    self.scroll_offset + content_height + 1,
                    lambda: self._confirm(stdscr,
                        f" Loaded {self.num_rows} rows — keep loading? "
                        f"May be slow / memory-heavy.   y = yes   n = stop "),
                )

            self.scroll_offset = max(0, min(self.scroll_offset, max(0, self.num_rows - 1)))
            if self.focused_col is not None:
                self.focused_col = max(0, min(self.focused_col, self.num_cols - 1))

            w = max_x - 1
            col_widths = self._get_column_widths(max_x)

            stdscr.erase()

            # Row 0: title bar (rows loaded / total, with a + while still loading)
            if self.total_rows is not None:
                rows_label = f"{self.num_rows}/{self.total_rows} rows"
            else:
                rows_label = f"{self.num_rows}{'' if self._fully_loaded else '+'} rows"
            title = f" {self.file_path.name} ({rows_label}, {self.num_cols} cols) "
            stdscr.addstr(0, 0, (title + "─" * max_x)[:w], stat_attr | curses.A_BOLD)

            # Row 1: top border
            stdscr.addstr(1, 0, self._render_separator(col_widths)[:w], data_attr)

            # Row 2: frozen column names (focused col highlighted)
            self._write_header_row(stdscr, 2, col_widths, w,
                                   col_attr, focus_hdr_attr, sep_attr)

            # Row 3 (optional): data types
            if self.show_dtypes:
                self._write_dtype_row_screen(stdscr, 3, col_widths, w,
                                             dtype_attr, focus_hdr_attr, sep_attr)

            # Bottom of header: separator
            stdscr.addstr(header_rows - 1, 0, self._render_separator(col_widths)[:w], data_attr)

            # Data rows — variable height when wrap is enabled
            screen_row = data_start
            row_idx = self.scroll_offset
            while screen_row < max_y - 1 and row_idx < self.num_rows:
                available = max_y - 1 - screen_row
                lines_used = self._write_data_row(
                    stdscr, screen_row, row_idx, col_widths, w,
                    data_attr, sep_attr, focus_data_attr,
                    max_screen_rows=available,
                )
                screen_row += lines_used
                row_idx += 1

            # Fill empty space below data
            while screen_row < max_y - 1:
                stdscr.addstr(screen_row, 0, "~"[:w], data_attr)
                screen_row += 1

            # Status bar
            thou_flag = "ON" if self.show_thousands else "OFF"
            type_flag = "ON" if self.show_dtypes else "OFF"
            tok_flag = "ON" if self.show_token_colors else "OFF"
            if self.focused_col is not None:
                fc_name = self.columns[self.focused_col]
                wrap_flag = "ON" if self.wrap_focused else "OFF"
                col_part = f"  col:{fc_name!r}  w:wrap[{wrap_flag}] |"
            else:
                col_part = ""
            if self.total_rows is not None:
                row_total = str(self.total_rows)
            else:
                row_total = str(self.num_rows) if self._fully_loaded else f"{self.num_rows}+"
            status = (
                f" Row {self.scroll_offset + 1}/{row_total}"
                f" | q:quit  ↑↓/PgDn:scroll  ←→:col  Esc:unfocus"
                f"{col_part}"
                f" | t:,sep[{thou_flag}]  d:types[{type_flag}]  c:tok[{tok_flag}]"
            )
            stdscr.addstr(max_y - 1, 0, status[:w], stat_attr)

            stdscr.noutrefresh()
            curses.doupdate()

            try:
                ch = stdscr.getch()
                if ch == ord('q'):
                    break
                elif ch == 27:                          # Esc — clear focus
                    self.focused_col = None
                    self.wrap_focused = False
                elif ch == curses.KEY_LEFT:
                    self._focus_left()
                elif ch == curses.KEY_RIGHT:
                    self._focus_right()
                elif ch == ord('w'):
                    if self.focused_col is not None:
                        self.wrap_focused = not self.wrap_focused
                elif ch == ord('t'):
                    self.show_thousands = not self.show_thousands
                elif ch == ord('d'):
                    self.show_dtypes = not self.show_dtypes
                elif ch == ord('c'):
                    self.show_token_colors = not self.show_token_colors
                elif ch == curses.KEY_UP:
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                elif ch == curses.KEY_DOWN:
                    # Let it grow past the loaded rows; the loop loads + clamps.
                    self._load_paused = False
                    self.scroll_offset += 1
                elif ch == curses.KEY_PPAGE:
                    self.scroll_offset = max(0, self.scroll_offset - content_height)
                elif ch == curses.KEY_NPAGE:
                    self._load_paused = False
                    self.scroll_offset += content_height
                elif ch == curses.KEY_HOME:
                    self.scroll_offset = 0
                elif ch == curses.KEY_END:
                    # Jump to the end of the currently-loaded rows (does not
                    # force-load the whole file).
                    self.scroll_offset = max(0, self.num_rows - content_height)
            except KeyboardInterrupt:
                break


def main():
    parser = argparse.ArgumentParser(
        description="View Parquet and Arrow files in a terminal UI (read-only)."
    )
    parser.add_argument("file", help="Path to .parquet or .arrow file")
    parser.add_argument("-n", "--rows", type=int, default=100,
                        help="Rows to load per chunk; more are loaded on demand "
                             "as you scroll (default: 100)")
    parser.add_argument("--load-limit", type=int, default=1000,
                        help="Prompt before loading beyond this many rows "
                             "(default: 1000)")
    args = parser.parse_args()

    try:
        t0 = datetime.now()
        print(f"[{t0.strftime('%Y-%m-%d %H:%M:%S')}] Opening {args.file}")
        viewer = DataViewer(args.file, max_rows=args.rows, load_limit=args.load_limit)
        print(f"Loaded first {viewer.num_rows} rows × {viewer.num_cols} cols "
              f"in {(datetime.now() - t0).total_seconds():.2f}s "
              f"(more load as you scroll)")
        curses.wrapper(viewer.display)
        viewer.close()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Viewer closed")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
