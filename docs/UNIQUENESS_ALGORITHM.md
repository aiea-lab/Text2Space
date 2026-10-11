# Uniqueness Inference Algorithm

## Core Concept

A spatial relationship between two objects is **uniquely inferable** if it can be deterministically computed from stated relations with no ambiguity.

### Compositional Approach

**Key Insight:** Any 2D spatial relation decomposes into orthogonal vertical and horizontal components.

```
"upper-left"  = "above"      ⊗ "left"
"below"       = "below"      ⊗ "same column"
"right"       = "same level" ⊗ "right"
```

**Theorem:** A relation is uniquely inferable **iff** both vertical AND horizontal components are uniquely inferable.

---

## High-Level Overview

### The Problem

Imagine you have objects arranged in 2D space with some stated relationships like "B is left of A" and "C is above B". The question is: can we **uniquely determine** the relationship between any two objects (like A and C) just from what's been stated?

Sometimes yes, sometimes no. The algorithm figures out which case we're in.

### The Solution Strategy

**Step 1: Simplify the Problem**

Instead of checking 8 possible spatial directions at once (upper-left, above, upper-right, etc.), we split the problem into two simpler questions:
- **Vertically:** Is it above, below, or at the same level?
- **Horizontally:** Is it left, right, or in the same column?

If we can uniquely answer both questions, we can uniquely determine the full spatial relation.

**Step 2: Follow the Paths**

Think of the stated relations as a graph where objects are nodes and relations are edges. To find the relationship between two objects:
1. Find **all shortest paths** between them through the graph
2. "Walk" along each path, composing the relations as you go
3. Check if all paths lead to the **same conclusion**

**Step 3: Watch for Traps**

There are two subtle ways ambiguity can sneak in:

**Trap 1: Opposing Directions**
- Say "B is above A" and "C is below B" and you want to know A vs C.
- Without knowing exact distances, we can't tell if the net result is above, below, or same level
- Solution: If any path contains opposing directions, mark it as ambiguous

**Trap 2: Cluster Ambiguity**
- Say "B is left of A" and "C is left of A", but nothing relates B and C
- B and C are "siblings" (same relation to a common reference)
- Solution: Check if siblings have an **independent path** between them (not going through their common reference)
- If no independent path exists, their relative position is unconstrained → ambiguous

**Step 4: Combine Results**

If both vertical and horizontal checks pass all validation steps, the relation is uniquely inferable. If either fails, it's ambiguous.

### Why This Works

The algorithm is comprehensive because it catches **all sources of ambiguity**:
- Objects in different disconnected components (no path exists)
- Multiple paths that disagree on the direction
- Paths with conflicting directions where magnitudes matter
- Groups of objects with identical relations to a reference but no mutual constraints

And it's efficient because:
- It shortcuts on direct relations (no need to search)
- It uses BFS to find only the shortest paths (minimal inference chains)
- It decomposes the 2D problem into two independent 1D problems

---

## Algorithm Structure

```python
is_relation_uniquely_inferable(obj1, obj2):
    return (is_vertical_relation_uniquely_inferable(obj1, obj2) AND
            is_horizontal_relation_uniquely_inferable(obj1, obj2))
```

Both component checks follow the same **5-step validation**:

### Five Steps (Per Component)

**1. Basic Validation**
```python
if obj1 not in objects or obj2 not in objects: return False
if obj1 == obj2: return True  # Same object = trivially unique
```

**2. Direct Relation Check**
```python
if (obj1, obj2) in edge_map: return True  # Stated relations are always unique
```

**3. Connectivity Check**
```python
if not are_objects_connected(obj1, obj2): return False  # No path = undefined
```

**4. Path Agreement Check**
```python
all_paths = find_all_shortest_paths(obj2, obj1)
inferred_component = check_path_agreement_component(all_paths, component_extractor)
if inferred_component is None: return False  # Paths disagree = ambiguous
```

**5. Cluster Ambiguity Check**
```python
if check_extended_cluster_ambiguity(obj1, obj2, inferred_component):
    return False  # Siblings without mutual constraints = ambiguous
```

---

## Key Implementation Details

### Path Agreement (`_check_path_agreement_component`)

For each path, accumulate component directions and check for conflicts:

```python
for path in all_paths:
    components = [extract_component(rel) for rel in path]

    # CRITICAL: Cannot assume unit magnitudes!
    # ["above", "below"] could be +5 above, -2 below = net +3 above

    if "above" in components and "below" in components:
        return None  # Opposing directions = unknown magnitude = ambiguous
    elif "above" in components:
        inferred.add("above")
    elif "below" in components:
        inferred.add("below")
    else:
        inferred.add("same level")

return inferred.pop() if len(inferred) == 1 else None  # Must all agree
```

**Why This Matters (Lines 1002-1009):**
- Naive composition assumes unit steps: `["above", "below"]` → cancel to `"same level"`
- Reality: magnitudes unknown → could be net `"above"`, `"below"`, or `"same level"`
- Solution: Detect opposing directions and mark as ambiguous

### Cluster Ambiguity (`_check_extended_cluster_ambiguity_component`)

**Problem:** Multiple objects with same relation to reference, but no mutual relation.

```python
# Find siblings: objects with same component relation to reference
siblings = find_siblings_with_same_component(obj1, reference, component)

for sibling in siblings:
    if not are_objects_connected(obj1, sibling):
        return True  # Ambiguous!

    if no_direct_edge(obj1, sibling):
        if not has_independent_path(obj1, sibling, excluded_node=reference):
            return True  # Only path goes through reference = ambiguous!

return False
```

**Independent Path Check:**
- If only path between siblings goes through reference, their mutual relation is unconstrained
- BFS from `obj1` to `sibling` with `reference` excluded from graph

---

## Correctness Proof

### Soundness: If algorithm returns True, relation is unique

**Proof by contrapositive:** Assume relation is ambiguous. Then one of these must hold:
1. Objects disconnected → Step 3 returns False ✓
2. Shortest paths disagree → Step 4 returns False ✓
3. Siblings lack constraints → Step 5 returns False ✓
4. Path has opposing directions → Step 4 returns False ✓

Algorithm catches all ambiguity sources.

### Completeness: If relation is unique, algorithm returns True

**Proof:** If relation is unique:
1. Objects must be connected → Step 3 passes ✓
2. All shortest paths must infer same component (by uniqueness) → Step 4 passes ✓
3. Any siblings must have mutual constraints (else relation wouldn't be unique) → Step 5 passes ✓

All checks pass.

### Compositional Correctness

8 spatial directions = Cartesian product of components:
```
{above, same_level, below} × {left, same_column, right}
```

- Components are orthogonal (independent in 2D)
- Both unique ⟹ product unique ✓
- Either ambiguous ⟹ product ambiguous ✓

---

## Examples

### ✓ Unique: Simple Chain
```
Relations: A-left-B, B-left-C
Query: A to C?

Path: A →(right)→ B →(right)→ C
  Horizontal: "right" + "right" = "right" ✓
  Vertical: "same" + "same" = "same" ✓
  No siblings ✓
Result: TRUE (C is right of A)
```

### ✗ Ambiguous: Cluster Without Constraints
```
Relations: B-upper_left-A, C-upper_left-A
Query: B to C?

Path: B →(lower_right)→ A →(upper_left)→ C
  Vertical: "below" + "above" = OPPOSING! ✗
Result: FALSE

(Alternative: B and C are vertical siblings w.r.t A, no independent path)
```

### ✓ Unique: Diamond Structure
```
Relations: B-above-A, C-above-A, D-above-B, D-above-C
Query: A to D?

Path 1: A →(above)→ B →(above)→ D
Path 2: A →(above)→ C →(above)→ D
  Both vertical: "above" ✓
  Siblings B,C have mutual constraint via D ✓
Result: TRUE (D is above A)
```

### ✓ Unique: L-Shape
```
Relations: B-above-A, C-right-B
Query: A to C?

Path: A →(above)→ B →(right)→ C
  Vertical: "above" ✓
  Horizontal: "right" ✓
  No siblings ✓
Result: TRUE (C is upper-right of A)
```

---

## Complexity

- **Per query:** O(V + E) — BFS for shortest paths dominates
- **Full validation:** O(V² × (V + E)) — check all O(V²) pairs
- **Space:** O(V + E) — adjacency list + edge map

**Optimizations:**
- Edge map for O(1) lookups (vs O(V) search)
- Lazy position computation
- Early termination at steps 1-3

---

## Summary

The algorithm is **sound and complete**:

✓ **Sound:** Reports unique only when truly unique
✓ **Complete:** Detects all unique relations
✓ **Efficient:** Linear per-query complexity
✓ **Compositional:** Elegant decomposition into orthogonal components

**Ambiguity Detection:**
1. Path disagreement (different shortest paths infer different components)
2. Opposing directions (single path contains both directions, unknown magnitude)
3. Cluster ambiguity (siblings with same relation to reference, no mutual constraint)
4. Disconnected components (no path exists)
