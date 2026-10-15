# docextract-schema/2

Input is JSON Schema 2020-12 or Draft 7, with an object root and explicit types. Without `$schema`,
2020-12 is selected. Declared dialects use their corresponding validator; original schemas are not
rewritten. Unknown dialects and nested dialect changes are rejected.
`schema-support` reports unsupported keyword paths before parsing or inference.

| Construct | Contract support |
| --- | --- |
| Named objects, nested objects, fixed properties | Supported |
| string, number, integer, boolean, null | Supported |
| Homogeneous object/primitive arrays | Supported |
| `type: [T, "null"]` | Supported |
| `anyOf`/`oneOf`: exactly one non-null branch plus `{ "type": "null" }` | Supported |
| Finite enum and const with explicit type | Supported and validated |
| Local, nonrecursive `#/$defs/...` references | Supported |
| required, additionalProperties boolean | Preserved and validated |
| Numeric, string, array and object size constraints | Preserved and validated |
| Formats registered by the locked jsonschema runtime | Validated; unknown formats rejected |
| External/recursive refs, constraint siblings of refs/unions | Unsupported |
| General unions, allOf, conditionals, dynamic/pattern properties, tuple arrays, boolean schemas | Unsupported |

Annotations such as titles, descriptions, examples and defaults remain in the original schema.
Defaults are annotations: validation never mutates data. With explicit `--apply-defaults`, B0 can
materialize a default on a missing planned leaf, labeled `schema_default`. Whole-object defaults
are not expanded. Competing mentions and printed-but-empty values do not become defaults.

Missing, observed null (`NULL` in the limited source grammar), zero, unreadable, ambiguous and
schema defaults remain distinct in field evidence. Absent fields are omitted from user data.
Missing required values make the result invalid/partial, never invented.

Schema support describes interpretation and validation, not universal extraction capability.
B0 additionally needs `bindings/1`: exact printed labels and table header names. It supports
root/nested object leaves, root tables, and one root section array with child tables. Deeper nested
record assembly is represented by the core contracts but is not discovered by this planner.
Primitive arrays must be printed as JSON in one bound field/cell.

Tables must have horizontal text, distinct left-aligned columns separated enough to produce
gap-derived header cells, explicit end labels, and a nonempty key column for new rows. Strings retain
case/punctuation and leading-zero identifiers; numeric parsing accepts simple decimal JSON numbers
without locale/currency guessing. Booleans must be lowercase `true`/`false`. No dates are reformatted.

Arbitrary PDFs, scans/handwriting, checkbox interpretation, complex headers and multimodal
interpretation remain unsupported. A learned schema-to-source proposal path is implemented separately;
its discovery/association quality is not established by the deterministic example or scripted tests.
Rotated/cropped source geometry is tested separately; the table planner uses native horizontal
coordinates and does not recognize arbitrary rotated glyph runs.

Internal paths are JSON Pointers. The ExtractBench adapter raises for property names containing
`.`/`[`/`]` or empty names that its dotted/bracketed grammar cannot safely represent.

The learned prototype reserves literal `*` property names for array scopes and rejects them before
inference. It validates all original constraints, accepts source spans for scalar/printed-primitive-array
leaves, and assembles discovered object arrays at arbitrary supported nesting depths. It does not infer
empty object arrays from missing proposals. Dates, currencies and locale-dependent numbers are not
automatically normalized. An input/output token limit saves incomplete work and fails explicitly;
automatic schema partitioning and unfinished-range subdivision remain work.

The learned regular-table path recognizes contiguous, fully populated native rows with matching
column geometry. The model must select a candidate and bind its header occurrences to schema fields.
Every candidate row is copied, preserving duplicate values. Repeated header-like rows are rejected
as ambiguous; missing cells, wrapped rows and complex headers are not generalized. Numeric parsing
rejects overflow or decimal values that cannot round-trip faithfully in the current JSON number
representation, leaving the field unresolved rather than silently rounding it.

The original remote audit found 70 distinct complete-schema hashes including annotations. Profile 1
accepted 66; four across 17 cases declared Draft 7. Profile 2 adds explicit Draft 7 validation. The
updated corpus audit must confirm coverage; schema support does not establish extraction accuracy
or faithful projection into a particular model template.

Validator selection follows the [locked jsonschema API](https://python-jsonschema.readthedocs.io/en/v4.23.0/validate/).
