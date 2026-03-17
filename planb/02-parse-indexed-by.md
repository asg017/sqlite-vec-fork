# Step 2: Parse `INDEXED BY ivf(...)` Syntax

*Reference: `plans/diskann-plans/02-parse-indexed-by.md` for the DiskANN equivalent.*

## Overview

Extend the column definition parser to recognize `INDEXED BY ivf(...)` and
populate `Vec0IvfConfig`. The existing scanner/tokenizer infrastructure (added
for DiskANN in `plans/prep-todo.md`) already supports LPAREN, RPAREN, COMMA
tokens.

## Parser Changes

In `vec0_parse_vector_column()`, after the existing DiskANN check:

```c
// Existing:
if (token == "diskann") { ... }
// New:
else if (token == "ivf") {
  rc = vec0_parse_ivf_options(scanner, &col->ivf);
}
```

## `vec0_parse_ivf_options()`

Parse key=value pairs inside `ivf(...)`:

```
ivf(nlist=256, nprobe=16)
```

The distance metric is NOT parsed here — it's already handled by the existing
`distance_metric=` column option in `vec0_parse_vector_column()`, which runs
before `INDEXED BY` parsing.

Implementation:

```c
static int vec0_parse_ivf_options(
  Vec0Scanner *scanner,
  struct Vec0IvfConfig *config
) {
  // Set defaults
  config->enabled = 1;
  config->nlist = VEC0_IVF_DEFAULT_NLIST;
  config->nprobe = VEC0_IVF_DEFAULT_NPROBE;

  // Expect LPAREN
  // Loop: parse key=value pairs separated by COMMA
  //   "nlist" => integer, validate range [0, 65536]
  //   "nprobe" => integer, validate range [1, 65536]
  // Expect RPAREN

  // Validation
  if (config->nlist > 0 && config->nprobe > config->nlist) {
    return error("nprobe must be <= nlist");
  }

  return SQLITE_OK;
}
```

## Validation

- Reject if both `diskann` and `ivf` are specified on the same column.
- `nlist=0` is valid (means "defer to compute-centroids").
- Unknown keys are errors.

## Test Cases

1. `float[128] indexed by ivf()` — all defaults
2. `float[128] indexed by ivf(nlist=256)` — custom nlist
3. `float[128] indexed by ivf(nlist=256, nprobe=32)` — both params
4. `float[128] indexed by ivf(nlist=0)` — deferred nlist
5. `float[128] indexed by ivf(nprobe=300, nlist=256)` — error: nprobe > nlist
6. `float[128] indexed by ivf(bogus=1)` — error: unknown key
7. `float[128] indexed by diskann() indexed by ivf()` — error: multiple indexes
8. `float[128] distance_metric=cosine indexed by ivf(nlist=64)` — metric from column, IVF just does clustering

## Files Changed

- `sqlite-vec-ivf.c`: Add `vec0_parse_ivf_options()`.
- `sqlite-vec.c`: ~3-line addition in `vec0_parse_vector_column()` to call
  `vec0_parse_ivf_options()` when token is `"ivf"` (same pattern as DiskANN
  dispatch).

## C Unit Tests

- Parser unit tests in the same pattern as DiskANN parser tests.
