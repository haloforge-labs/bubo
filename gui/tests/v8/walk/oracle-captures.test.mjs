import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { oracle } from "./oracle.mjs";

for (const [state, scenario] of [
  ["S7", "stopped-saved-estimate"],
  ["S8", "memory-gpu-headroom"],
  ["S9", "memory-default-setup"],
  ["S19", "memory-smaller-context"],
  ["S20", "models-warnings"],
]) {
  test(`${state} walk expectation matches the current CLI-derived scenario`, () => {
    const view = JSON.parse(
      readFileSync(
        new URL(
          `../../../wails/frontend/src/scenarios/${scenario}.json`,
          import.meta.url,
        ),
        "utf8",
      ),
    );
    assert.equal(oracle[state].fit, view.fit.sentence);
  });
}
