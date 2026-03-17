#include "../sqlite-vec.h"
#include "sqlite-vec-internal.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>

#define countof(x) (sizeof(x) / sizeof((x)[0]))

// Tests vec0_token_next(), the low-level tokenizer that extracts the next
// token from a raw char range. Covers every token type (identifier, digit,
// brackets, plus, equals), whitespace skipping, EOF on empty/whitespace-only
// input, error on unrecognised characters, and boundary behaviour where
// identifiers and digits stop at the next non-matching character.
void test_vec0_token_next() {
  printf("Starting %s...\n", __func__);
  struct Vec0Token token;
  int rc;
  char *input;

  // Single-character tokens
  input = "+";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_PLUS);

  input = "[";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_LBRACKET);

  input = "]";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_RBRACKET);

  input = "=";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_EQ);

  // Identifier
  input = "hello";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
  assert(token.start == input);
  assert(token.end == input + 5);

  // Identifier with underscores and digits
  input = "col_1a";
  rc = vec0_token_next(input, input + 6, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
  assert(token.end - token.start == 6);

  // Digit sequence
  input = "1234";
  rc = vec0_token_next(input, input + 4, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_DIGIT);
  assert(token.start == input);
  assert(token.end == input + 4);

  // Leading whitespace is skipped
  input = "  abc";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
  assert(token.end - token.start == 3);

  // Tab/newline whitespace
  input = "\t\n\r X";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);

  // Empty input
  input = "";
  rc = vec0_token_next(input, input, &token);
  assert(rc == VEC0_TOKEN_RESULT_EOF);

  // Only whitespace
  input = "   ";
  rc = vec0_token_next(input, input + 3, &token);
  assert(rc == VEC0_TOKEN_RESULT_EOF);

  // Unrecognized character
  input = "@";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_ERROR);

  input = "!";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_ERROR);

  // Identifier stops at bracket
  input = "foo[";
  rc = vec0_token_next(input, input + 4, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
  assert(token.end - token.start == 3);

  // Digit stops at non-digit
  input = "42abc";
  rc = vec0_token_next(input, input + 5, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_DIGIT);
  assert(token.end - token.start == 2);

  // Left paren
  input = "(";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_LPAREN);

  // Right paren
  input = ")";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_RPAREN);

  // Comma
  input = ",";
  rc = vec0_token_next(input, input + 1, &token);
  assert(rc == VEC0_TOKEN_RESULT_SOME);
  assert(token.token_type == TOKEN_TYPE_COMMA);

  printf("  All vec0_token_next tests passed.\n");
}

// Tests Vec0Scanner, the stateful wrapper around vec0_token_next() that
// tracks position and yields successive tokens. Verifies correct tokenisation
// of full sequences like "abc float[128]" and "key=value", empty input,
// whitespace-heavy input, and expressions with operators ("a+b").
void test_vec0_scanner() {
  printf("Starting %s...\n", __func__);
  struct Vec0Scanner scanner;
  struct Vec0Token token;
  int rc;

  // Scan "abc float[128]"
  {
    const char *input = "abc float[128]";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(token.end - token.start == 3);
    assert(strncmp(token.start, "abc", 3) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(token.end - token.start == 5);
    assert(strncmp(token.start, "float", 5) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_LBRACKET);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_DIGIT);
    assert(strncmp(token.start, "128", 3) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_RBRACKET);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan "key=value"
  {
    const char *input = "key=value";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "key", 3) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_EQ);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "value", 5) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan empty string
  {
    const char *input = "";
    vec0_scanner_init(&scanner, input, 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan with lots of whitespace
  {
    const char *input = "  a   b  ";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(token.end - token.start == 1);
    assert(*token.start == 'a');

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(token.end - token.start == 1);
    assert(*token.start == 'b');

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan "a+b"
  {
    const char *input = "a+b";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_PLUS);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  // Scan "myindex(k=v, k2=v2)"
  {
    const char *input = "myindex(k=v, k2=v2)";
    vec0_scanner_init(&scanner, input, (int)strlen(input));

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "myindex", 7) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_LPAREN);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "k", 1) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_EQ);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "v", 1) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_COMMA);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "k2", 2) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_EQ);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_IDENTIFIER);
    assert(strncmp(token.start, "v2", 2) == 0);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_SOME);
    assert(token.token_type == TOKEN_TYPE_RPAREN);

    rc = vec0_scanner_next(&scanner, &token);
    assert(rc == VEC0_TOKEN_RESULT_EOF);
  }

  printf("  All vec0_scanner tests passed.\n");
}

// Tests vec0_parse_vector_column(), which parses a vec0 column definition
// string like "embedding float[768] distance_metric=cosine" into a
// VectorColumnDefinition struct. Covers all element types (float/f32, int8/i8,
// bit), column names with underscores/digits, all distance metrics (L2, L1,
// cosine), the default metric, and error cases: empty input, missing type,
// unknown type, missing dimensions, unknown metric, unknown option key, and
// distance_metric on bit columns.
void test_vec0_parse_vector_column() {
  printf("Starting %s...\n", __func__);
  struct VectorColumnDefinition col;
  int rc;

  // Basic float column
  {
    const char *input = "embedding float[768]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.name_length == 9);
    assert(strncmp(col.name, "embedding", 9) == 0);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    assert(col.dimensions == 768);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_L2);
    sqlite3_free(col.name);
  }

  // f32 alias
  {
    const char *input = "v f32[3]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_FLOAT32);
    assert(col.dimensions == 3);
    sqlite3_free(col.name);
  }

  // int8 column
  {
    const char *input = "quantized int8[256]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_INT8);
    assert(col.dimensions == 256);
    assert(col.name_length == 9);
    assert(strncmp(col.name, "quantized", 9) == 0);
    sqlite3_free(col.name);
  }

  // i8 alias
  {
    const char *input = "q i8[64]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_INT8);
    assert(col.dimensions == 64);
    sqlite3_free(col.name);
  }

  // bit column
  {
    const char *input = "bvec bit[1024]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_BIT);
    assert(col.dimensions == 1024);
    sqlite3_free(col.name);
  }

  // Column name with underscores and digits
  {
    const char *input = "col_name_2 float[10]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.name_length == 10);
    assert(strncmp(col.name, "col_name_2", 10) == 0);
    sqlite3_free(col.name);
  }

  // distance_metric=cosine
  {
    const char *input = "emb float[128] distance_metric=cosine";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    assert(col.dimensions == 128);
    sqlite3_free(col.name);
  }

  // distance_metric=L2 (explicit)
  {
    const char *input = "emb float[128] distance_metric=L2";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_L2);
    sqlite3_free(col.name);
  }

  // distance_metric=L1
  {
    const char *input = "emb float[128] distance_metric=l1";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_L1);
    sqlite3_free(col.name);
  }

  // SQLITE_EMPTY: empty string
  {
    const char *input = "";
    rc = vec0_parse_vector_column(input, 0, &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: non-vector column (text primary key)
  {
    const char *input = "document_id text primary key";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: non-vector column (partition key)
  {
    const char *input = "user_id integer partition key";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: no type (single identifier)
  {
    const char *input = "emb";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: unknown type
  {
    const char *input = "emb double[128]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: unknown type (unknowntype)
  {
    const char *input = "v unknowntype[128]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // SQLITE_EMPTY: missing brackets entirely
  {
    const char *input = "emb float";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_EMPTY);
  }

  // Error: zero dimensions
  {
    const char *input = "v float[0]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: empty brackets (no dimensions)
  {
    const char *input = "v float[]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown distance metric
  {
    const char *input = "emb float[128] distance_metric=hamming";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown distance metric (foo)
  {
    const char *input = "v float[128] distance_metric=foo";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown option key
  {
    const char *input = "emb float[128] foobar=baz";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: distance_metric on bit type
  {
    const char *input = "emb bit[64] distance_metric=cosine";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // indexed by flat()
  {
    const char *input = "emb float[768] indexed by flat()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_FLAT);
    assert(col.dimensions == 768);
    sqlite3_free(col.name);
  }

  // indexed by flat() with distance_metric
  {
    const char *input = "emb float[768] distance_metric=cosine indexed by flat()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_FLAT);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    sqlite3_free(col.name);
  }

  // indexed by flat() on int8
  {
    const char *input = "emb int8[256] indexed by flat()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_FLAT);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_INT8);
    sqlite3_free(col.name);
  }

  // indexed by flat() on bit
  {
    const char *input = "emb bit[64] indexed by flat()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_FLAT);
    assert(col.element_type == SQLITE_VEC_ELEMENT_TYPE_BIT);
    sqlite3_free(col.name);
  }

  // default index_type is FLAT
  {
    const char *input = "emb float[768]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_FLAT);
    sqlite3_free(col.name);
  }

  // Error: indexed by (missing type name)
  {
    const char *input = "emb float[768] indexed by";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: indexed by unknown()
  {
    const char *input = "emb float[768] indexed by unknown()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: indexed by flat (missing parens)
  {
    const char *input = "emb float[768] indexed by flat";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: indexed flat() (missing "by")
  {
    const char *input = "emb float[768] indexed flat()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // IVF: indexed by ivf() — defaults
  {
    const char *input = "v float[4] indexed by ivf()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_IVF);
    assert(col.dimensions == 4);
    assert(col.index_type == VEC0_INDEX_TYPE_IVF);
    assert(col.ivf.nlist == 128);  // default
    assert(col.ivf.nprobe == 10);  // default
    sqlite3_free(col.name);
  }

  // IVF: indexed by ivf(nlist=8) — nprobe auto-clamped to 8
  {
    const char *input = "v float[4] indexed by ivf(nlist=8)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_IVF);
    assert(col.index_type == VEC0_INDEX_TYPE_IVF);
    assert(col.ivf.nlist == 8);
    assert(col.ivf.nprobe == 8);  // clamped from default 10
    sqlite3_free(col.name);
  }

  // IVF: indexed by ivf(nlist=64, nprobe=8)
  {
    const char *input = "v float[4] indexed by ivf(nlist=64, nprobe=8)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_IVF);
    assert(col.ivf.nlist == 64);
    assert(col.ivf.nprobe == 8);
    sqlite3_free(col.name);
  }

  // IVF: with distance_metric before indexed by
  {
    const char *input = "v float[4] distance_metric=cosine indexed by ivf(nlist=16)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_IVF);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    assert(col.index_type == VEC0_INDEX_TYPE_IVF);
    assert(col.ivf.nlist == 16);
    sqlite3_free(col.name);
  }

  // IVF: nlist=0 (deferred)
  {
    const char *input = "v float[4] indexed by ivf(nlist=0)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.ivf.nlist == 0);
    sqlite3_free(col.name);
  }

  // IVF error: nprobe > nlist
  {
    const char *input = "v float[4] indexed by ivf(nlist=4, nprobe=10)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // IVF error: unknown key
  {
    const char *input = "v float[4] indexed by ivf(bogus=1)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // IVF error: unknown index type (hnsw not supported)
  {
    const char *input = "v float[4] indexed by hnsw()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Not IVF: no ivf config
  {
    const char *input = "v float[4]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_FLAT);
    sqlite3_free(col.name);
  }

  // IVF: quantizer=binary
  {
    const char *input = "v float[768] indexed by ivf(nlist=128, quantizer=binary)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_IVF);
    assert(col.ivf.nlist == 128);
    assert(col.ivf.quantizer == VEC0_IVF_QUANTIZER_BINARY);
    assert(col.ivf.oversample == 1);
    sqlite3_free(col.name);
  }

  // IVF: quantizer=int8
  {
    const char *input = "v float[768] indexed by ivf(nlist=64, quantizer=int8)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.ivf.quantizer == VEC0_IVF_QUANTIZER_INT8);
    sqlite3_free(col.name);
  }

  // IVF: quantizer=none (explicit)
  {
    const char *input = "v float[768] indexed by ivf(quantizer=none)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.ivf.quantizer == VEC0_IVF_QUANTIZER_NONE);
    sqlite3_free(col.name);
  }

  // IVF: oversample=10 with quantizer
  {
    const char *input = "v float[768] indexed by ivf(nlist=128, quantizer=binary, oversample=10)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.ivf.quantizer == VEC0_IVF_QUANTIZER_BINARY);
    assert(col.ivf.oversample == 10);
    assert(col.ivf.nlist == 128);
    sqlite3_free(col.name);
  }

  // IVF: all params
  {
    const char *input = "v float[768] distance_metric=cosine indexed by ivf(nlist=256, nprobe=16, quantizer=int8, oversample=4)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    assert(col.ivf.nlist == 256);
    assert(col.ivf.nprobe == 16);
    assert(col.ivf.quantizer == VEC0_IVF_QUANTIZER_INT8);
    assert(col.ivf.oversample == 4);
    sqlite3_free(col.name);
  }

  // IVF error: oversample > 1 without quantizer
  {
    const char *input = "v float[768] indexed by ivf(oversample=10)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // IVF error: unknown quantizer value
  {
    const char *input = "v float[768] indexed by ivf(quantizer=pq)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // IVF: quantizer with defaults (nlist=128 default, nprobe=10 default)
  {
    const char *input = "v float[768] indexed by ivf(quantizer=binary, oversample=5)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.ivf.nlist == 128);
    assert(col.ivf.nprobe == 10);
    assert(col.ivf.quantizer == VEC0_IVF_QUANTIZER_BINARY);
    assert(col.ivf.oversample == 5);
    sqlite3_free(col.name);
  }

  printf("  All vec0_parse_vector_column tests passed.\n");
}

// Tests vec0_parse_partition_key_definition(), which parses a vec0 partition
// key column definition like "user_id integer partition key". Verifies correct
// parsing of integer and text partition keys, column name extraction, and
// rejection of invalid inputs: empty strings, non-partition-key definitions
// ("primary key"), and misspelled keywords.
void test_vec0_parse_partition_key_definition() {
  printf("Starting %s...\n", __func__);
  typedef struct {
    char * test;
    int expected_rc;
    const char *expected_column_name;
    int expected_column_type;
  } TestCase;

  TestCase suite[] = {
    {"user_id integer partition key", SQLITE_OK, "user_id", SQLITE_INTEGER},
    {"USER_id int partition key", SQLITE_OK, "USER_id", SQLITE_INTEGER},
    {"category text partition key", SQLITE_OK, "category", SQLITE_TEXT},

    {"", SQLITE_EMPTY, "", 0},
    {"document_id text primary key", SQLITE_EMPTY, "", 0},
    {"document_id text partition keyy", SQLITE_EMPTY, "", 0},
  };
  for(int i = 0; i < countof(suite); i++) {
    char * out_column_name;
    int out_column_name_length;
    int out_column_type;
    int rc;
    rc = vec0_parse_partition_key_definition(
      suite[i].test,
      strlen(suite[i].test),
      &out_column_name,
      &out_column_name_length,
      &out_column_type
    );
    assert(rc == suite[i].expected_rc);

    if(rc == SQLITE_OK) {
      assert(out_column_name_length == strlen(suite[i].expected_column_name));
      assert(strncmp(out_column_name, suite[i].expected_column_name, out_column_name_length) == 0);
      assert(out_column_type == suite[i].expected_column_type);
    }

    printf("  Passed: \"%s\"\n", suite[i].test);
  }
}

void test_distance_l2_sqr_float() {
  printf("Starting %s...\n", __func__);
  float d;

  // Identical vectors: distance = 0
  {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    d = _test_distance_l2_sqr_float(a, b, 3);
    assert(d == 0.0f);
  }

  // Orthogonal unit vectors: sqrt(1+1) = sqrt(2)
  {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    d = _test_distance_l2_sqr_float(a, b, 3);
    assert(fabsf(d - sqrtf(2.0f)) < 1e-6f);
  }

  // Known computation: [1,2,3] vs [4,5,6] = sqrt(9+9+9) = sqrt(27)
  {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    d = _test_distance_l2_sqr_float(a, b, 3);
    assert(fabsf(d - sqrtf(27.0f)) < 1e-5f);
  }

  // Single dimension: sqrt(16) = 4.0
  {
    float a[] = {3.0f};
    float b[] = {7.0f};
    d = _test_distance_l2_sqr_float(a, b, 1);
    assert(d == 4.0f);
  }

  printf("  All distance_l2_sqr_float tests passed.\n");
}

void test_distance_cosine_float() {
  printf("Starting %s...\n", __func__);
  float d;

  // Identical direction: distance = 0.0
  {
    float a[] = {1.0f, 0.0f};
    float b[] = {2.0f, 0.0f};
    d = _test_distance_cosine_float(a, b, 2);
    assert(fabsf(d - 0.0f) < 1e-6f);
  }

  // Orthogonal: distance = 1.0
  {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    d = _test_distance_cosine_float(a, b, 2);
    assert(fabsf(d - 1.0f) < 1e-6f);
  }

  // Opposite direction: distance = 2.0
  {
    float a[] = {1.0f, 0.0f};
    float b[] = {-1.0f, 0.0f};
    d = _test_distance_cosine_float(a, b, 2);
    assert(fabsf(d - 2.0f) < 1e-6f);
  }

  printf("  All distance_cosine_float tests passed.\n");
}

void test_distance_hamming() {
  printf("Starting %s...\n", __func__);
  float d;

  // Identical bitmaps: distance = 0
  {
    unsigned char a[] = {0xFF};
    unsigned char b[] = {0xFF};
    d = _test_distance_hamming(a, b, 8);
    assert(d == 0.0f);
  }

  // All different: distance = 8
  {
    unsigned char a[] = {0xFF};
    unsigned char b[] = {0x00};
    d = _test_distance_hamming(a, b, 8);
    assert(d == 8.0f);
  }

  // Half different: 0xFF vs 0x0F = 4 bits differ
  {
    unsigned char a[] = {0xFF};
    unsigned char b[] = {0x0F};
    d = _test_distance_hamming(a, b, 8);
    assert(d == 4.0f);
  }

  // Multi-byte: [0xFF, 0x00] vs [0x00, 0xFF] = 16 bits differ
  {
    unsigned char a[] = {0xFF, 0x00};
    unsigned char b[] = {0x00, 0xFF};
    d = _test_distance_hamming(a, b, 16);
    assert(d == 16.0f);
  }

  printf("  All distance_hamming tests passed.\n");
}

void test_vec0_parse_vector_column_diskann() {
  printf("Starting %s...\n", __func__);
  struct VectorColumnDefinition col;
  int rc;

  // Existing syntax (no INDEXED BY) should have diskann.enabled == 0
  {
    const char *input = "emb float[128]";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type != VEC0_INDEX_TYPE_DISKANN);
    sqlite3_free(col.name);
  }

  // With distance_metric but no INDEXED BY
  {
    const char *input = "emb float[128] distance_metric=cosine";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type != VEC0_INDEX_TYPE_DISKANN);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    sqlite3_free(col.name);
  }

  // Basic binary quantizer
  {
    const char *input = "emb float[128] INDEXED BY diskann(neighbor_quantizer=binary)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_DISKANN);
    assert(col.diskann.quantizer_type == VEC0_DISKANN_QUANTIZER_BINARY);
    assert(col.diskann.n_neighbors == 72);  // default
    assert(col.diskann.search_list_size == 128);  // default
    assert(col.dimensions == 128);
    sqlite3_free(col.name);
  }

  // INT8 quantizer
  {
    const char *input = "v float[64] INDEXED BY diskann(neighbor_quantizer=int8)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_DISKANN);
    assert(col.diskann.quantizer_type == VEC0_DISKANN_QUANTIZER_INT8);
    sqlite3_free(col.name);
  }

  // Custom n_neighbors
  {
    const char *input = "emb float[128] INDEXED BY diskann(neighbor_quantizer=binary, n_neighbors=48)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_DISKANN);
    assert(col.diskann.n_neighbors == 48);
    sqlite3_free(col.name);
  }

  // Custom search_list_size
  {
    const char *input = "emb float[128] INDEXED BY diskann(neighbor_quantizer=binary, search_list_size=256)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.diskann.search_list_size == 256);
    sqlite3_free(col.name);
  }

  // Combined with distance_metric (distance_metric first)
  {
    const char *input = "emb float[128] distance_metric=cosine INDEXED BY diskann(neighbor_quantizer=int8)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.distance_metric == VEC0_DISTANCE_METRIC_COSINE);
    assert(col.index_type == VEC0_INDEX_TYPE_DISKANN);
    assert(col.diskann.quantizer_type == VEC0_DISKANN_QUANTIZER_INT8);
    sqlite3_free(col.name);
  }

  // Error: missing neighbor_quantizer (required)
  {
    const char *input = "emb float[128] INDEXED BY diskann(n_neighbors=72)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: empty parens
  {
    const char *input = "emb float[128] INDEXED BY diskann()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown quantizer
  {
    const char *input = "emb float[128] INDEXED BY diskann(neighbor_quantizer=unknown)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: bad n_neighbors (not divisible by 8)
  {
    const char *input = "emb float[128] INDEXED BY diskann(neighbor_quantizer=binary, n_neighbors=13)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: n_neighbors too large
  {
    const char *input = "emb float[128] INDEXED BY diskann(neighbor_quantizer=binary, n_neighbors=512)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: missing BY
  {
    const char *input = "emb float[128] INDEXED diskann(neighbor_quantizer=binary)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown algorithm
  {
    const char *input = "emb float[128] INDEXED BY hnsw(neighbor_quantizer=binary)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Error: unknown option key
  {
    const char *input = "emb float[128] INDEXED BY diskann(neighbor_quantizer=binary, foobar=baz)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_ERROR);
  }

  // Case insensitivity for keywords
  {
    const char *input = "emb float[128] indexed by DISKANN(NEIGHBOR_QUANTIZER=BINARY)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == SQLITE_OK);
    assert(col.index_type == VEC0_INDEX_TYPE_DISKANN);
    assert(col.diskann.quantizer_type == VEC0_DISKANN_QUANTIZER_BINARY);
    sqlite3_free(col.name);
  }

  printf("  All vec0_parse_vector_column_diskann tests passed.\n");
}

void test_diskann_validity_bitmap() {
  printf("Starting %s...\n", __func__);

  unsigned char validity[3]; // 24 bits
  memset(validity, 0, sizeof(validity));

  // All initially invalid
  for (int i = 0; i < 24; i++) {
    assert(diskann_validity_get(validity, i) == 0);
  }
  assert(diskann_validity_count(validity, 24) == 0);

  // Set bit 0
  diskann_validity_set(validity, 0, 1);
  assert(diskann_validity_get(validity, 0) == 1);
  assert(diskann_validity_count(validity, 24) == 1);

  // Set bit 7 (last bit of first byte)
  diskann_validity_set(validity, 7, 1);
  assert(diskann_validity_get(validity, 7) == 1);
  assert(diskann_validity_count(validity, 24) == 2);

  // Set bit 8 (first bit of second byte)
  diskann_validity_set(validity, 8, 1);
  assert(diskann_validity_get(validity, 8) == 1);
  assert(diskann_validity_count(validity, 24) == 3);

  // Set bit 23 (last bit)
  diskann_validity_set(validity, 23, 1);
  assert(diskann_validity_get(validity, 23) == 1);
  assert(diskann_validity_count(validity, 24) == 4);

  // Clear bit 0
  diskann_validity_set(validity, 0, 0);
  assert(diskann_validity_get(validity, 0) == 0);
  assert(diskann_validity_count(validity, 24) == 3);

  // Other bits unaffected
  assert(diskann_validity_get(validity, 7) == 1);
  assert(diskann_validity_get(validity, 8) == 1);

  printf("  All diskann_validity_bitmap tests passed.\n");
}

void test_diskann_neighbor_ids() {
  printf("Starting %s...\n", __func__);

  unsigned char ids[8 * 8]; // 8 slots * 8 bytes each
  memset(ids, 0, sizeof(ids));

  // Set and get slot 0
  diskann_neighbor_id_set(ids, 0, 42);
  assert(diskann_neighbor_id_get(ids, 0) == 42);

  // Set and get middle slot
  diskann_neighbor_id_set(ids, 3, 12345);
  assert(diskann_neighbor_id_get(ids, 3) == 12345);

  // Set and get last slot
  diskann_neighbor_id_set(ids, 7, 99999);
  assert(diskann_neighbor_id_get(ids, 7) == 99999);

  // Slot 0 still correct
  assert(diskann_neighbor_id_get(ids, 0) == 42);

  // Large value
  diskann_neighbor_id_set(ids, 1, INT64_MAX);
  assert(diskann_neighbor_id_get(ids, 1) == INT64_MAX);

  printf("  All diskann_neighbor_ids tests passed.\n");
}

void test_diskann_quantize_binary() {
  printf("Starting %s...\n", __func__);

  // 8-dimensional vector: positive values -> 1, negative/zero -> 0
  float src[8] = {1.0f, -1.0f, 0.5f, 0.0f, -0.5f, 0.1f, -0.1f, 100.0f};
  unsigned char out[1]; // 8 bits = 1 byte

  int rc = diskann_quantize_vector(src, 8, VEC0_DISKANN_QUANTIZER_BINARY, out);
  assert(rc == 0);

  // Expected bits (LSB first within each byte):
  // bit 0: 1.0 > 0 -> 1
  // bit 1: -1.0 > 0 -> 0
  // bit 2: 0.5 > 0 -> 1
  // bit 3: 0.0 > 0 -> 0  (not strictly greater)
  // bit 4: -0.5 > 0 -> 0
  // bit 5: 0.1 > 0 -> 1
  // bit 6: -0.1 > 0 -> 0
  // bit 7: 100.0 > 0 -> 1
  // Expected byte: 1 + 0 + 4 + 0 + 0 + 32 + 0 + 128 = 0b10100101 = 0xA5
  assert(out[0] == 0xA5);

  printf("  All diskann_quantize_binary tests passed.\n");
}

void test_diskann_node_init_sizes() {
  printf("Starting %s...\n", __func__);

  unsigned char *validity, *ids, *qvecs;
  int validitySize, idsSize, qvecsSize;

  // 72 neighbors, binary quantizer, 1024 dims
  int rc = diskann_node_init(72, VEC0_DISKANN_QUANTIZER_BINARY, 1024,
      &validity, &validitySize, &ids, &idsSize, &qvecs, &qvecsSize);
  assert(rc == 0);
  assert(validitySize == 9);      // 72/8
  assert(idsSize == 576);         // 72 * 8
  assert(qvecsSize == 9216);      // 72 * (1024/8)

  // All validity bits should be 0
  assert(diskann_validity_count(validity, 72) == 0);

  sqlite3_free(validity);
  sqlite3_free(ids);
  sqlite3_free(qvecs);

  // 8 neighbors, int8 quantizer, 32 dims
  rc = diskann_node_init(8, VEC0_DISKANN_QUANTIZER_INT8, 32,
      &validity, &validitySize, &ids, &idsSize, &qvecs, &qvecsSize);
  assert(rc == 0);
  assert(validitySize == 1);    // 8/8
  assert(idsSize == 64);        // 8 * 8
  assert(qvecsSize == 256);     // 8 * 32

  sqlite3_free(validity);
  sqlite3_free(ids);
  sqlite3_free(qvecs);

  printf("  All diskann_node_init_sizes tests passed.\n");
}

void test_diskann_node_set_clear_neighbor() {
  printf("Starting %s...\n", __func__);

  unsigned char *validity, *ids, *qvecs;
  int validitySize, idsSize, qvecsSize;

  // 8 neighbors, binary quantizer, 16 dims (2 bytes per qvec)
  int rc = diskann_node_init(8, VEC0_DISKANN_QUANTIZER_BINARY, 16,
      &validity, &validitySize, &ids, &idsSize, &qvecs, &qvecsSize);
  assert(rc == 0);

  // Create a test quantized vector (2 bytes)
  unsigned char test_qvec[2] = {0xAB, 0xCD};

  // Set neighbor at slot 3
  diskann_node_set_neighbor(validity, ids, qvecs, 3,
      42, test_qvec, VEC0_DISKANN_QUANTIZER_BINARY, 16);

  // Verify slot 3 is valid
  assert(diskann_validity_get(validity, 3) == 1);
  assert(diskann_validity_count(validity, 8) == 1);

  // Verify rowid
  assert(diskann_neighbor_id_get(ids, 3) == 42);

  // Verify quantized vector
  const unsigned char *read_qvec = diskann_neighbor_qvec_get(
      qvecs, 3, VEC0_DISKANN_QUANTIZER_BINARY, 16);
  assert(read_qvec[0] == 0xAB);
  assert(read_qvec[1] == 0xCD);

  // Clear slot 3
  diskann_node_clear_neighbor(validity, ids, qvecs, 3,
      VEC0_DISKANN_QUANTIZER_BINARY, 16);
  assert(diskann_validity_get(validity, 3) == 0);
  assert(diskann_neighbor_id_get(ids, 3) == 0);
  assert(diskann_validity_count(validity, 8) == 0);

  sqlite3_free(validity);
  sqlite3_free(ids);
  sqlite3_free(qvecs);

  printf("  All diskann_node_set_clear_neighbor tests passed.\n");
}

void test_diskann_prune_select() {
  printf("Starting %s...\n", __func__);

  // Scenario: 5 candidates, sorted by distance to p
  // Candidates: A(0), B(1), C(2), D(3), E(4)
  // p_distances (already sorted): A=1.0, B=2.0, C=3.0, D=4.0, E=5.0
  //
  // Inter-candidate distances (5x5 matrix):
  //       A     B     C     D     E
  // A   0.0   1.5   3.0   4.0   5.0
  // B   1.5   0.0   1.5   3.0   4.0
  // C   3.0   1.5   0.0   1.5   3.0
  // D   4.0   3.0   1.5   0.0   1.5
  // E   5.0   4.0   3.0   1.5   0.0

  float p_distances[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float inter[25] = {
    0.0f, 1.5f, 3.0f, 4.0f, 5.0f,
    1.5f, 0.0f, 1.5f, 3.0f, 4.0f,
    3.0f, 1.5f, 0.0f, 1.5f, 3.0f,
    4.0f, 3.0f, 1.5f, 0.0f, 1.5f,
    5.0f, 4.0f, 3.0f, 1.5f, 0.0f,
  };
  int selected[5];
  int count;

  // alpha=1.0, R=3: greedy selection
  // Round 1: Pick A (closest). Prune check:
  //   B: 1.0*1.5 <= 2.0? yes -> pruned
  //   C: 1.0*3.0 <= 3.0? yes -> pruned
  //   D: 1.0*4.0 <= 4.0? yes -> pruned
  //   E: 1.0*5.0 <= 5.0? yes -> pruned
  // Result: only A selected
  {
    int rc = diskann_prune_select(inter, p_distances, 5, 1.0f, 3, selected, &count);
    assert(rc == 0);
    assert(count == 1);
    assert(selected[0] == 1); // A
  }

  // alpha=1.5, R=3: diversity-aware
  // Round 1: Pick A. Prune check:
  //   B: 1.5*1.5=2.25 <= 2.0? no -> keep
  //   C: 1.5*3.0=4.5 <= 3.0? no -> keep
  //   D: 1.5*4.0=6.0 <= 4.0? no -> keep
  //   E: 1.5*5.0=7.5 <= 5.0? no -> keep
  // Round 2: Pick B. Prune check:
  //   C: 1.5*1.5=2.25 <= 3.0? yes -> pruned
  //   D: 1.5*3.0=4.5 <= 4.0? no -> keep
  //   E: 1.5*4.0=6.0 <= 5.0? no -> keep
  // Round 3: Pick D. Done, 3 selected.
  {
    int rc = diskann_prune_select(inter, p_distances, 5, 1.5f, 3, selected, &count);
    assert(rc == 0);
    assert(count == 3);
    assert(selected[0] == 1); // A
    assert(selected[1] == 1); // B
    assert(selected[3] == 1); // D
    assert(selected[2] == 0); // C pruned
    assert(selected[4] == 0); // E not reached
  }

  // R > num_candidates with very high alpha (no pruning): select all
  {
    int rc = diskann_prune_select(inter, p_distances, 5, 100.0f, 10, selected, &count);
    assert(rc == 0);
    assert(count == 5);
  }

  // Empty candidate set
  {
    int rc = diskann_prune_select(NULL, NULL, 0, 1.2f, 3, selected, &count);
    assert(rc == 0);
    assert(count == 0);
  }

  printf("  All diskann_prune_select tests passed.\n");
}

void test_diskann_quantized_vector_byte_size() {
  printf("Starting %s...\n", __func__);

  // Binary quantizer: 1 bit per dimension, so 128 dims = 16 bytes
  assert(diskann_quantized_vector_byte_size(VEC0_DISKANN_QUANTIZER_BINARY, 128) == 16);
  assert(diskann_quantized_vector_byte_size(VEC0_DISKANN_QUANTIZER_BINARY, 8) == 1);
  assert(diskann_quantized_vector_byte_size(VEC0_DISKANN_QUANTIZER_BINARY, 1024) == 128);

  // INT8 quantizer: 1 byte per dimension
  assert(diskann_quantized_vector_byte_size(VEC0_DISKANN_QUANTIZER_INT8, 128) == 128);
  assert(diskann_quantized_vector_byte_size(VEC0_DISKANN_QUANTIZER_INT8, 1) == 1);
  assert(diskann_quantized_vector_byte_size(VEC0_DISKANN_QUANTIZER_INT8, 768) == 768);

  printf("  All diskann_quantized_vector_byte_size tests passed.\n");
}

void test_diskann_config_defaults() {
  printf("Starting %s...\n", __func__);

  // A freshly zero-initialized VectorColumnDefinition should have diskann.enabled == 0
  struct VectorColumnDefinition col;
  memset(&col, 0, sizeof(col));
  assert(col.index_type != VEC0_INDEX_TYPE_DISKANN);
  assert(col.diskann.n_neighbors == 0);
  assert(col.diskann.search_list_size == 0);

  // Verify parsing a normal vector column still works and diskann is not enabled
  {
    const char *input = "embedding float[768]";
    int rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == 0 /* SQLITE_OK */);
    assert(col.index_type != VEC0_INDEX_TYPE_DISKANN);
    sqlite3_free(col.name);
  }

  printf("  All diskann_config_defaults tests passed.\n");
}

// ============================================================
// Annoy tests
// ============================================================

void test_annoy_config_defaults() {
  printf("Starting %s...\n", __func__);
  struct VectorColumnDefinition col;
  memset(&col, 0, sizeof(col));

  // Verify parsing a normal vector column has annoy disabled
  {
    const char *input = "embedding float[768]";
    int rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == 0 /* SQLITE_OK */);
    assert(col.index_type != VEC0_INDEX_TYPE_ANNOY);
    assert(col.annoy.n_trees == 0);
    assert(col.annoy.search_k == 0);
    sqlite3_free(col.name);
  }

  // Verify parsing a normal column with distance_metric has annoy disabled
  {
    const char *input = "emb float[128] distance_metric=cosine";
    int rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == 0 /* SQLITE_OK */);
    assert(col.index_type != VEC0_INDEX_TYPE_ANNOY);
    sqlite3_free(col.name);
  }

  printf("  All annoy_config_defaults tests passed.\n");
}

void test_vec0_parse_vector_column_annoy() {
  printf("Starting %s...\n", __func__);
  struct VectorColumnDefinition col;
  int rc;

  // Basic annoy with defaults
  {
    const char *input = "embedding float[768] INDEXED BY annoy()";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == 0 /* SQLITE_OK */);
    assert(col.index_type == VEC0_INDEX_TYPE_ANNOY);
    assert(col.annoy.n_trees == 50);
    assert(col.annoy.search_k == 0);
    assert(col.dimensions == 768);
    sqlite3_free(col.name);
  }

  // Custom n_trees
  {
    const char *input = "emb float[384] INDEXED BY annoy(n_trees=100)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == 0 /* SQLITE_OK */);
    assert(col.index_type == VEC0_INDEX_TYPE_ANNOY);
    assert(col.annoy.n_trees == 100);
    assert(col.annoy.search_k == 0);
    sqlite3_free(col.name);
  }

  // Custom search_k
  {
    const char *input = "emb float[384] INDEXED BY annoy(n_trees=10, search_k=500)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == 0 /* SQLITE_OK */);
    assert(col.index_type == VEC0_INDEX_TYPE_ANNOY);
    assert(col.annoy.n_trees == 10);
    assert(col.annoy.search_k == 500);
    sqlite3_free(col.name);
  }

  // With distance_metric
  {
    const char *input = "emb float[768] distance_metric=cosine INDEXED BY annoy(n_trees=25)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == 0 /* SQLITE_OK */);
    assert(col.index_type == VEC0_INDEX_TYPE_ANNOY);
    assert(col.annoy.n_trees == 25);
    assert(col.distance_metric == 2 /* VEC0_DISTANCE_METRIC_COSINE */);
    sqlite3_free(col.name);
  }

  // int8 element type
  {
    const char *input = "emb int8[768] INDEXED BY annoy(n_trees=50)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc == 0 /* SQLITE_OK */);
    assert(col.index_type == VEC0_INDEX_TYPE_ANNOY);
    sqlite3_free(col.name);
  }

  // Error: n_trees too large
  {
    const char *input = "emb float[768] INDEXED BY annoy(n_trees=9999)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc != 0);
  }

  // Error: n_trees = 0
  {
    const char *input = "emb float[768] INDEXED BY annoy(n_trees=0)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc != 0);
  }

  // Error: unknown option
  {
    const char *input = "emb float[768] INDEXED BY annoy(bad_option=5)";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc != 0);
  }

  // Error: missing parens
  {
    const char *input = "emb float[768] INDEXED BY annoy";
    rc = vec0_parse_vector_column(input, (int)strlen(input), &col);
    assert(rc != 0);
  }

  printf("  All vec0_parse_vector_column_annoy tests passed.\n");
}

void test_annoy_node_encode_decode() {
  printf("Starting %s...\n", __func__);

  // Test split node encode/decode (float32, no quantization)
  {
    float split_vec[4] = {1.0f, -0.5f, 0.0f, 0.25f};
    float query[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    unsigned char *data = NULL;
    int dataSize = 0;
    int rc = annoy_encode_split_node(42, 99, split_vec, 4,
                                      VEC0_ANNOY_QUANTIZER_NONE, &data, &dataSize);
    assert(rc == 0 /* SQLITE_OK */);
    assert(dataSize == 2 * 4 + 4 * 4);  // 2 ints + 4 floats = 24 bytes
    assert(data != NULL);

    int left, right;
    float margin;
    rc = annoy_decode_split_node(data, dataSize, 4,
                                  VEC0_ANNOY_QUANTIZER_NONE,
                                  &left, &right, query, &margin);
    assert(rc == 0);
    assert(left == 42);
    assert(right == 99);
    // margin = dot([1,1,1,1], [1,-0.5,0,0.25]) = 1 - 0.5 + 0 + 0.25 = 0.75
    assert(margin > 0.74f && margin < 0.76f);

    sqlite3_free(data);
  }

  // Test split node encode/decode with int8 quantization
  {
    float split_vec[4] = {1.0f, -0.5f, 0.0f, 0.25f};
    float query[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    unsigned char *data = NULL;
    int dataSize = 0;
    int rc = annoy_encode_split_node(42, 99, split_vec, 4,
                                      VEC0_ANNOY_QUANTIZER_INT8, &data, &dataSize);
    assert(rc == 0);
    assert(dataSize == 2 * 4 + 4 * 1);  // 2 ints + 4 int8s = 12 bytes
    assert(data != NULL);

    int left, right;
    float margin;
    rc = annoy_decode_split_node(data, dataSize, 4,
                                  VEC0_ANNOY_QUANTIZER_INT8,
                                  &left, &right, query, &margin);
    assert(rc == 0);
    assert(left == 42);
    assert(right == 99);
    // Should be approximately 0.75 (with quantization error)
    assert(margin > 0.5f && margin < 1.0f);

    sqlite3_free(data);
  }

  // Test split node encode/decode with binary quantization
  {
    float split_vec[8] = {1.0f, -0.5f, 0.3f, -0.1f, 0.8f, -0.2f, 0.0f, 0.5f};
    float query[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    unsigned char *data = NULL;
    int dataSize = 0;
    int rc = annoy_encode_split_node(10, 20, split_vec, 8,
                                      VEC0_ANNOY_QUANTIZER_BINARY, &data, &dataSize);
    assert(rc == 0);
    assert(dataSize == 2 * 4 + 1);  // 2 ints + 1 byte (8 bits)
    assert(data != NULL);

    int left, right;
    float margin;
    rc = annoy_decode_split_node(data, dataSize, 8,
                                  VEC0_ANNOY_QUANTIZER_BINARY,
                                  &left, &right, query, &margin);
    assert(rc == 0);
    assert(left == 10);
    assert(right == 20);
    assert(margin > 1.5f && margin < 2.5f);

    sqlite3_free(data);
  }

  // Test descendants node encode/decode
  {
    long long rowids[5] = {10, 20, 30, 40, 50};
    unsigned char *data = NULL;
    int dataSize = 0;
    int rc = annoy_encode_descendants_node(rowids, 5, &data, &dataSize);
    assert(rc == 0);
    assert(dataSize == 5 * 8);  // 5 * sizeof(i64)
    assert(data != NULL);

    const long long *decoded_rowids;
    int count;
    rc = annoy_decode_descendants_node(data, dataSize, &decoded_rowids, &count);
    assert(rc == 0);
    assert(count == 5);
    assert(decoded_rowids[0] == 10);
    assert(decoded_rowids[4] == 50);

    sqlite3_free(data);
  }

  // Test empty descendants
  {
    unsigned char *data = NULL;
    int dataSize = 0;
    int rc = annoy_encode_descendants_node(NULL, 0, &data, &dataSize);
    assert(rc == 0);
    assert(dataSize == 0);
    sqlite3_free(data);
  }

  // Test decode with wrong size
  {
    unsigned char buf[3] = {0, 0, 0};  // not divisible by 8
    const long long *rowids;
    int count;
    int rc = annoy_decode_descendants_node(buf, 3, &rowids, &count);
    assert(rc != 0);  // should error
  }

  printf("  All annoy_node_encode_decode tests passed.\n");
}

int main() {
  printf("Starting unit tests...\n");
#ifdef SQLITE_VEC_ENABLE_AVX
  printf("SQLITE_VEC_ENABLE_AVX=1\n");
#endif
#ifdef SQLITE_VEC_ENABLE_NEON
  printf("SQLITE_VEC_ENABLE_NEON=1\n");
#endif
#if !defined(SQLITE_VEC_ENABLE_AVX) && !defined(SQLITE_VEC_ENABLE_NEON)
  printf("SIMD: none\n");
#endif
  test_vec0_token_next();
  test_vec0_scanner();
  test_vec0_parse_vector_column();
  test_vec0_parse_partition_key_definition();
  test_distance_l2_sqr_float();
  test_distance_cosine_float();
  test_distance_hamming();
  test_vec0_parse_vector_column_diskann();
  test_diskann_validity_bitmap();
  test_diskann_neighbor_ids();
  test_diskann_quantize_binary();
  test_diskann_node_init_sizes();
  test_diskann_node_set_clear_neighbor();
  test_diskann_prune_select();
  test_diskann_quantized_vector_byte_size();
  test_diskann_config_defaults();
  test_annoy_config_defaults();
  test_vec0_parse_vector_column_annoy();
  test_annoy_node_encode_decode();
  printf("All unit tests passed.\n");
}
