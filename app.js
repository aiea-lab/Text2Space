// Opening puzzle: guess from the text, sketch on a grid, read the answer off the sketch,
// then see the same sketch as an ASCII grid. Every interaction is optional.

const GRID_SIZE = 7;
const DEMO_STEP_MS = 650;

// Offsets are [dx, dy] with y increasing downward. A relation depends only on the signs.
const DIRECTIONS = {
  "upper-left": [-1, -1],
  above: [0, -1],
  "upper-right": [1, -1],
  left: [-1, 0],
  right: [1, 0],
  "lower-left": [-1, 1],
  below: [0, 1],
  "lower-right": [1, 1],
};

const PHRASES = {
  "upper-left": "to the upper-left of",
  above: "directly above",
  "upper-right": "to the upper-right of",
  left: "directly left of",
  right: "directly right of",
  "lower-left": "to the lower-left of",
  below: "directly below",
  "lower-right": "to the lower-right of",
};

const ARROWS = {
  "upper-left": "↖",
  above: "↑",
  "upper-right": "↗",
  left: "←",
  right: "→",
  "lower-left": "↙",
  below: "↓",
  "lower-right": "↘",
};

// Hour each direction points to on the clock dial beside the description.
const CLOCK_HOURS = {
  above: 12,
  "upper-right": 1.5,
  right: 3,
  "lower-right": 4.5,
  below: 6,
  "lower-left": 7.5,
  left: 9,
  "upper-left": 10.5,
};

const DIAL_NUMBER_RADIUS = 38; // in the dial's 100-unit viewBox

// Row-major 3x3 layout of the answer compass; null is the reference place in the middle.
const COMPASS_LAYOUT = [
  "upper-left", "above", "upper-right",
  "left", null, "right",
  "lower-left", "below", "lower-right",
];

const MARKS = { pending: "○", ok: "✓", bad: "✗" };
const STATUS_LABELS = { pending: "not drawn yet", ok: "matches your map", bad: "does not match your map" };

const { places, sentences, query, solution } = PUZZLE;

// Places in order of first mention; the tray hands them out in this order.
const PLACE_ORDER = [...new Set(sentences.flatMap((s) => [s.a, s.b]))];

// The first place mentioned starts on the board, in the middle, so the reader has a fixed point.
const ANCHOR = PLACE_ORDER[0];
const BOARD_CENTER = Math.floor(GRID_SIZE / 2);

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);

const state = {
  positions: {}, // place id -> [x, y]
  selected: null, // place the next tap on the board will put down or move
  guess: null, // direction picked on the compass
  drawnByReader: false, // false while the board holds only the page's own drawing
  activeSentence: null, // sentence the demo is currently drawing
  demoTimer: null,
};

// ---------- spatial logic ----------

function relationBetween(posA, posB) {
  const dx = Math.sign(posA[0] - posB[0]);
  const dy = Math.sign(posA[1] - posB[1]);
  return Object.keys(DIRECTIONS).find((d) => DIRECTIONS[d][0] === dx && DIRECTIONS[d][1] === dy);
}

function sentenceStatus(sentence) {
  const posA = state.positions[sentence.a];
  const posB = state.positions[sentence.b];
  if (!posA || !posB) return { status: "pending" };
  const drawn = relationBetween(posA, posB);
  return { status: drawn === sentence.dir ? "ok" : "bad", drawn };
}

// True when the place sits where one of its sentences says it should not.
function isMisplaced(id) {
  return sentences.some((s) => (s.a === id || s.b === id) && sentenceStatus(s).status === "bad");
}

function isSolved() {
  return sentences.every((s) => sentenceStatus(s).status === "ok");
}

function bounds(layout) {
  const xs = Object.values(layout).map((p) => p[0]);
  const ys = Object.values(layout).map((p) => p[1]);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

// Crop a layout to its bounding box: rows of place ids, null for empty cells.
function toRows(layout) {
  const { minX, maxX, minY, maxY } = bounds(layout);
  const rows = Array.from({ length: maxY - minY + 1 }, () => Array(maxX - minX + 1).fill(null));
  for (const [id, [x, y]] of Object.entries(layout)) rows[y - minY][x - minX] = id;
  return rows;
}

// Same format as the dataset's `grid` style: "+--+" borders, one letter per cell.
function toAsciiLines(rows) {
  const border = "+" + "--+".repeat(rows[0].length);
  const lines = [border];
  for (const row of rows) {
    lines.push("|" + row.map((id) => (id ?? " ") + " |").join(""), border);
  }
  return lines;
}

// Keep one free row or column next to the sketch while the board has room for it.
function recenter() {
  const { minX, maxX, minY, maxY } = bounds(state.positions);
  const shift = (min, max) => {
    if (min === 0 && max < GRID_SIZE - 1) return 1;
    if (max === GRID_SIZE - 1 && min > 0) return -1;
    return 0;
  };
  const dx = shift(minX, maxX);
  const dy = shift(minY, maxY);
  if (!dx && !dy) return;
  for (const id of Object.keys(state.positions)) {
    const [x, y] = state.positions[id];
    state.positions[id] = [x + dx, y + dy];
  }
}

// ---------- text helpers ----------

function placeLabel(id, withEmoji) {
  const { name, emoji } = places[id];
  return `<span class="place">${withEmoji ? `${emoji}&nbsp;` : ""}${name}</span>`;
}

// Expand a sentence template: {X} becomes a place, [phrase] the emphasized direction words.
// With `dialDir`, the direction words also become a control that points the clock dial.
function renderText(text, { withEmoji = false, dialDir = null } = {}) {
  const control = dialDir ? ` data-dir="${dialDir}" role="button" tabindex="0"` : "";
  return text
    .replace(/\{([A-Z])\}/g, (_, id) => placeLabel(id, withEmoji))
    .replace(/\[(.+?)\]/g, (_, phrase) => `<span class="dir"${control}>${phrase}</span>`);
}

function placeAt(x, y) {
  return Object.keys(state.positions).find((id) => {
    const [px, py] = state.positions[id];
    return px === x && py === y;
  });
}

function nextUnplaced() {
  return PLACE_ORDER.find((id) => !state.positions[id]) ?? null;
}

// ---------- one-time construction ----------

function buildStatic() {
  $("story").innerHTML = sentences.map((s) => renderText(s.text, { dialDir: s.dir })).join(" ");
  $("question").innerHTML = renderText(query.text);

  $("dial-hours").innerHTML = Array.from({ length: 12 }, (_, i) => {
    const hour = i + 1;
    const angle = (hour * Math.PI) / 6;
    const x = (DIAL_NUMBER_RADIUS * Math.sin(angle)).toFixed(1);
    const y = (-DIAL_NUMBER_RADIUS * Math.cos(angle)).toFixed(1);
    return `<text x="${x}" y="${y}"${hour % 3 === 0 ? ' class="is-major"' : ""}>${hour}</text>`;
  }).join("");

  $("compass").innerHTML = COMPASS_LAYOUT.map((dir) =>
    dir
      ? `<button type="button" class="compass-option" data-dir="${dir}" aria-pressed="false">
           <span class="compass-arrow" aria-hidden="true">${ARROWS[dir]}</span>
           <span class="compass-label">${dir}</span>
         </button>`
      : `<div class="compass-center"><span aria-hidden="true">${places[query.b].emoji}</span>${places[query.b].name}</div>`,
  ).join("");

  $("tray").innerHTML = PLACE_ORDER.map(
    (id) =>
      `<button type="button" class="chip" data-id="${id}" aria-pressed="false">
         <span aria-hidden="true">${places[id].emoji}</span>${places[id].name}
       </button>`,
  ).join("");

  $("cells").innerHTML = Array.from({ length: GRID_SIZE * GRID_SIZE }, (_, i) => {
    const x = i % GRID_SIZE;
    const y = Math.floor(i / GRID_SIZE);
    return `<button type="button" class="cell" data-x="${x}" data-y="${y}"></button>`;
  }).join("");

  $("tokens").innerHTML = PLACE_ORDER.map(
    (id) => `<div class="token" data-id="${id}" hidden><span>${places[id].emoji}</span></div>`,
  ).join("");

  $("key").innerHTML = Object.keys(places)
    .sort()
    .map((id) => `<li><code>${id}</code> ${placeLabel(id, true)}</li>`)
    .join("");
}

// ---------- rendering ----------

function renderGuess(solved) {
  for (const button of $("compass").querySelectorAll(".compass-option")) {
    const dir = button.dataset.dir;
    button.setAttribute("aria-pressed", String(dir === state.guess));
    button.classList.toggle("is-answer", solved && dir === query.answer);
  }
  $("guess-note").textContent = state.guess
    ? `You said ${state.guess}. Hold on to that answer. The verdict comes after the sketch.`
    : "";
  $("guess-note").hidden = !state.guess || solved;
}

function renderTray() {
  for (const chip of $("tray").querySelectorAll(".chip")) {
    const id = chip.dataset.id;
    chip.setAttribute("aria-pressed", String(id === state.selected));
    chip.classList.toggle("is-placed", Boolean(state.positions[id]));
  }
}

function renderBoard(solved) {
  $("board").classList.toggle("is-solved", solved);
  $("board").classList.toggle("is-holding", Boolean(state.selected));

  for (const token of $("tokens").children) {
    const id = token.dataset.id;
    const pos = state.positions[id];
    token.hidden = !pos;
    if (pos) {
      token.style.setProperty("--x", pos[0]);
      token.style.setProperty("--y", pos[1]);
    }
    token.classList.toggle("is-selected", id === state.selected);
    token.classList.toggle("is-queried", solved && (id === query.a || id === query.b));
  }

  for (const cell of $("cells").children) {
    const occupant = placeAt(Number(cell.dataset.x), Number(cell.dataset.y));
    const where = `column ${Number(cell.dataset.x) + 1}, row ${Number(cell.dataset.y) + 1}`;
    cell.setAttribute("aria-label", occupant ? `${places[occupant].name}, ${where}` : `empty, ${where}`);
  }

  const line = $("sightline").querySelector("line");
  $("sightline").classList.toggle("is-visible", solved);
  if (solved) {
    const [fromX, fromY] = state.positions[query.b];
    const [toX, toY] = state.positions[query.a];
    line.setAttribute("x1", fromX + 0.5);
    line.setAttribute("y1", fromY + 0.5);
    line.setAttribute("x2", toX + 0.5);
    line.setAttribute("y2", toY + 0.5);
  }
}

// A one-line tally next to the board, so a phone reader need not scroll up to the checklist.
function renderProgress(statuses) {
  const ok = statuses.filter((s) => s.status === "ok").length;
  const bad = statuses.filter((s) => s.status === "bad").length;
  $("progress").hidden = ok + bad === 0;
  $("progress").innerHTML =
    `${ok} of ${sentences.length} sentences match your map` +
    (bad ? `, <span class="drawn">${bad} ${bad === 1 ? "does" : "do"} not</span>.` : ".");
}

function renderChecklist() {
  const statuses = sentences.map(sentenceStatus);
  renderProgress(statuses);
  $("checklist").innerHTML = sentences
    .map((sentence, i) => {
      const { status, drawn } = statuses[i];
      const mismatch =
        status === "bad"
          ? `<span class="drawn">On your map the ${places[sentence.a].name} is ${PHRASES[drawn]} the ${places[sentence.b].name}.</span>`
          : "";
      const active = i === state.activeSentence ? " is-active" : "";
      return `<li class="is-${status}${active}">
                <span class="mark" aria-hidden="true">${MARKS[status]}</span>
                <span class="sentence">${renderText(sentence.text, { withEmoji: true })}${mismatch}
                  <span class="sr-only">(${STATUS_LABELS[status]})</span></span>
              </li>`;
    })
    .join("");
}

function renderReveal(solved) {
  $("reveal").hidden = !solved;
  if (!solved) return;
  const drawn = relationBetween(state.positions[query.a], state.positions[query.b]);
  const answer = `The ${places[query.a].name} is ${PHRASES[drawn]} the ${places[query.b].name}`;
  let recall = ".";
  if (state.guess === drawn) recall = ", as you said.";
  else if (state.guess) recall = `. In your head you said ${state.guess}.`;
  $("reveal").innerHTML = `<strong>${answer}${recall}</strong> With the map drawn, the answer is there to read.`;
}

let bridgeKey = null;

// The bridge shows the reader's own sketch once it is complete, the reference sketch otherwise.
function renderBridge(solved) {
  const own = solved && state.drawnByReader;
  const layout = own ? state.positions : solution;
  const key = JSON.stringify([own, layout]);
  if (key === bridgeKey) return;
  bridgeKey = key;

  const rows = toRows(layout);
  $("minimap").style.setProperty("--cols", rows[0].length);
  $("minimap").innerHTML = rows
    .flat()
    .map((id) => `<span>${id ? places[id].emoji : ""}</span>`)
    .join("");
  $("ascii").innerHTML = toAsciiLines(rows)
    .map((line, i) => `<span class="line" style="--i:${i}">${line}</span>`)
    .join("");
  $("translation-caption").textContent = own
    ? "Your sketch as an ASCII grid."
    : "The same map as an ASCII grid.";
}

function update() {
  const solved = isSolved();
  renderGuess(solved);
  renderTray();
  renderBoard(solved);
  renderChecklist();
  renderReveal(solved);
  renderBridge(solved);
}

// ---------- interaction ----------

// Turn the dial's hand to the direction a phrase in the description stands for.
function pointDial(phrase) {
  for (const other of $("story").querySelectorAll(".dir")) {
    other.classList.toggle("is-active", other === phrase);
  }
  $("dial-hand").style.setProperty("--hour", CLOCK_HOURS[phrase.dataset.dir]);
}

function stopDemo() {
  clearTimeout(state.demoTimer);
  state.demoTimer = null;
  state.activeSentence = null;
}

// Back to the starting board: only the anchor placed, the next place in hand.
function clearBoard() {
  stopDemo();
  state.positions = { [ANCHOR]: [BOARD_CENTER, BOARD_CENTER] };
  state.selected = nextUnplaced();
  state.drawnByReader = false;
}

// Draw the reference sketch sentence by sentence around the anchor.
function startDemo() {
  clearBoard();
  state.selected = null;

  const offsetX = BOARD_CENTER - solution[ANCHOR][0];
  const offsetY = BOARD_CENTER - solution[ANCHOR][1];
  const target = (id) => [solution[id][0] + offsetX, solution[id][1] + offsetY];

  if (reducedMotion) {
    for (const id of PLACE_ORDER) state.positions[id] = target(id);
    update();
    return;
  }

  // One step per newly placed landmark; a sentence that adds none still gets its turn.
  const steps = [];
  const drawn = new Set([ANCHOR]);
  sentences.forEach((sentence, i) => {
    const fresh = [sentence.b, sentence.a].filter((id) => !drawn.has(id));
    fresh.forEach((id) => drawn.add(id));
    (fresh.length ? fresh : [null]).forEach((id) => steps.push({ sentence: i, id }));
  });

  const run = (k) => {
    if (k === steps.length) {
      stopDemo();
      update();
      return;
    }
    const { sentence, id } = steps[k];
    state.activeSentence = sentence;
    if (id) state.positions[id] = target(id);
    update();
    state.demoTimer = setTimeout(() => run(k + 1), DEMO_STEP_MS);
  };
  run(0);
}

function handleCell(x, y) {
  stopDemo();
  const occupant = placeAt(x, y);
  if (occupant) {
    state.selected = state.selected === occupant ? null : occupant;
  } else if (state.selected) {
    const id = state.selected;
    state.positions[id] = [x, y];
    state.drawnByReader = true;
    // Slide the sketch off an edge only while places remain that may need the room.
    if (nextUnplaced()) recenter();
    // A misplaced place stays in hand, so the next tap moves it instead of placing another.
    if (!isMisplaced(id)) state.selected = nextUnplaced();
  }
  update();
}

function bindEvents() {
  for (const type of ["pointerover", "focusin", "click"]) {
    $("story").addEventListener(type, (event) => {
      const phrase = event.target.closest(".dir");
      if (phrase) pointDial(phrase);
    });
  }

  $("compass").addEventListener("click", (event) => {
    const button = event.target.closest(".compass-option");
    if (!button) return;
    state.guess = state.guess === button.dataset.dir ? null : button.dataset.dir;
    update();
  });

  $("tray").addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    stopDemo();
    state.selected = state.selected === chip.dataset.id ? null : chip.dataset.id;
    update();
  });

  $("cells").addEventListener("click", (event) => {
    const cell = event.target.closest(".cell");
    if (cell) handleCell(Number(cell.dataset.x), Number(cell.dataset.y));
  });

  $("demo").addEventListener("click", startDemo);
  $("clear").addEventListener("click", () => {
    clearBoard();
    update();
  });

  // Write the ASCII grid out line by line the first time it scrolls into view.
  const observer = new IntersectionObserver(
    (entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      $("translation").classList.add("is-in-view");
      observer.disconnect();
    },
    { threshold: 0.4 },
  );
  observer.observe($("translation"));
}

buildStatic();
bindEvents();
clearBoard();
// Start the dial on the least familiar clock position in the description.
pointDial($("story").querySelector('.dir[data-dir="upper-left"]'));
update();
