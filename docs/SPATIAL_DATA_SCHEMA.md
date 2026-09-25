# Text2Space Data Schema

`spatial_data.jsonl` holds one JSON object per line. Each object is a synthetic 2D layout of 2 to 8 objects named `A`, `B`, `C`, ..., a natural-language description of some of the relations between them, one query about a pair of objects, and the ground-truth layout in three ASCII styles. The released file has 20,000 instances; the paper evaluates on the 1,000 listed in `spatial_data_tested.jsonl`.

## Example

```json
{
  "id": "id-1",
  "description": "A is to the right of D. B is to the north and to the west of A. C is at the 1 o'clock position from A. D is at the bottom-left of C. D is southward from B.",
  "query_relation": "Where is A relative to D?",
  "query_type": "full",
  "label": "right",
  "detailed_relations": "A is below and to the right of B. A is below and to the left of C. A is to the right of D. ...",
  "ascii": {
    "simple": "B     C\nD  A",
    "grid": "+--+--+--+\n|B |  |C |\n+--+--+--+\n|D |A |  |\n+--+--+--+",
    "panel": "┌──────────┐\n│ B     C  │\n│ D  A     │\n└──────────┘"
  },
  "terminology_used": "spatial+cardinal+clock",
  "num_components": 4,
  "num_relations": 5,
  "is_directly_stated": true,
  "has_unique_layout": false,
  "ambiguous_stages": 4
}
```

## Fields

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Unique instance id, `id-1` ... `id-20000`. |
| `description` | string | The relations the model is given, one sentence per stated relation. Only the `num_relations` relations used to build the layout appear here; every other pair must be inferred. |
| `query_relation` | string | The question about one ordered pair: "Where is A relative to D?" (full), "Where is A vertically relative to D?" (vertical), "Where is A horizontally relative to D?" (horizontal). |
| `query_type` | `full`, `vertical`, `horizontal` | Which component of the relation the query asks for. |
| `label` | string | Ground-truth answer. Full queries: one of `above`, `below`, `left`, `right`, `upper-left`, `upper-right`, `lower-left`, `lower-right`. Vertical queries: `above`, `below`, `same level`. Horizontal queries: `left`, `right`, `same column`. |
| `detailed_relations` | string | Every ordered pair's relation as realized by the ground-truth layout, in base terminology. Used to score description generation (task B). |
| `ascii` | object | The ground-truth layout rendered as `simple` (objects separated by spaces), `grid` (ASCII table with `+--+` borders), and `panel` (Unicode box frame). All experiments in the paper use `grid`. |
| `terminology_used` | string | Which vocabularies the description draws on, joined by `+`: `spatial` (above, left of, ...), `cardinal` (north, southeast, ...), `clock` (at 3 o'clock from, ...). |
| `num_components` | int, 2-8 | Number of objects in the layout. |
| `num_relations` | int, 1-12 | Number of relations stated in the description. Every layout is connected, so `num_relations >= num_components - 1`. |
| `is_directly_stated` | bool | Whether the queried pair's relation is one of the stated relations (`true`) or must be inferred by composing stated relations (`false`). |
| `has_unique_layout` | bool | Whether every pairwise relation in the layout is uniquely determined by the stated relations. See `UNIQUENESS_ALGORITHM.md`. |
| `ambiguous_stages` | int, 0-11 | Number of relations after whose addition the partial layout was not uniquely determined. A later relation can resolve an earlier ambiguity, so a layout with `has_unique_layout = true` may still have a positive count. |

The current generator additionally writes `reasoning_steps` (a chain of stated relations and intermediate compositions that derives the answer). The released file predates that field; `scripts/backfill_reasoning_steps.py` computes it for an existing file.

## Coordinates and directions

Positions are `(x, y)` on an integer grid with `y` increasing downward, matching row indices in the ASCII renderings. `above` means smaller `y`, `left` means smaller `x`, and a diagonal such as `upper-left` means both. Relations are stored in both directions: stating "B is above A" also records "A is below B".

Every relation decomposes into a vertical component (`above`, `below`, `same level`) and a horizontal component (`left`, `right`, `same column`). Vertical and horizontal queries ask for one component; full queries ask for the pair, which is one of the eight direction labels. The generator never asks a full query about a pair that shares a row or a column, so `same level` and `same column` occur only as answers to vertical and horizontal queries.

## How instances are generated

1. Draw the number of objects (2-8) and connect them one relation at a time: pick a placed object, an unplaced object, and a random direction, then add the relation. After all objects are connected, extra relations between placed objects may be added. `num_relations` counts every relation added.
2. Realize the layout on the grid and render the three ASCII styles.
3. Pick a query pair and query type. Compute the answer from the realized positions. Mark `is_directly_stated` by checking whether that pair's relation was added in step 1.
4. Write the description from the stated relations only, choosing the vocabularies recorded in `terminology_used`.
5. Run the uniqueness check over all pairs to set `has_unique_layout`, and record `ambiguous_stages` from the placement trace.

Generation is balanced with rejection sampling so that component counts, relation counts, query types, terminology mixes, direct versus inferred queries, and unique versus ambiguous layouts are each spread across their ranges. `src/data/spatial_data_generator.py --help` lists the targets.
