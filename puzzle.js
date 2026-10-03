// The opening puzzle: Text2Space instance id-3943, with objects A-F shown as town landmarks.
// Coordinates are [x, y] with y increasing downward, as in the dataset's ASCII grids.

const PUZZLE = {
  id: "id-3943",

  places: {
    A: { name: "café", emoji: "☕️" },
    B: { name: "library", emoji: "📚" },
    C: { name: "park", emoji: "🌳" },
    D: { name: "station", emoji: "🚉" },
    E: { name: "cinema", emoji: "🎬" },
    F: { name: "bakery", emoji: "🥖" },
  },

  // Each sentence states where `a` is relative to `b`.
  // In the text, {X} marks a place and [phrase] marks the words that carry the direction.
  sentences: [
    { a: "F", b: "D", dir: "below", text: "The {F} is [below] the {D}." },
    { a: "C", b: "D", dir: "left", text: "The {C} is at the [9:00] position from the {D}." },
    { a: "A", b: "F", dir: "below", text: "The {A} is at [6 o'clock] from the {F}." },
    { a: "E", b: "C", dir: "upper-left", text: "The {E} is at [10:30] from the {C}." },
    { a: "B", b: "E", dir: "above", text: "The {B} is [above] the {E}." },
    { a: "C", b: "F", dir: "upper-left", text: "The {C} is [above and to the left] of the {F}." },
  ],

  query: { a: "B", b: "F", answer: "upper-left", text: "Where is the {B} relative to the {F}?" },

  solution: { B: [0, 0], E: [0, 1], C: [1, 2], D: [2, 2], F: [2, 3], A: [2, 4] },
};
