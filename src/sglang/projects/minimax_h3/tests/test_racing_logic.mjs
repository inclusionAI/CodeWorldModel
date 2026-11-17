import assert from 'node:assert/strict';
import {createRace, ROAD_HALF_WIDTH, stepRace, trackPose, TRACK_LENGTH} from '../web/playable/racing-logic.mjs';

const advance = (race, seconds, controls) => {
  for (let time = 0; time < seconds - 1e-9; time += .1) stepRace(race, Math.min(.1, seconds - time), controls);
  return race;
};
const near = (actual, expected, tolerance = 1e-6) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

const race = createRace();
const initial = structuredClone(race);
stepRace(race, 1, {});
assert.deepEqual(race, initial, 'ready pose does not drift');
for (const dt of [0, -1, NaN, Infinity]) stepRace(race, dt, {});
assert.deepEqual(race, initial);
stepRace(race, 0, {throttle: true});
assert.equal(race.phase, 'countdown');
assert.equal(race.countdown, 3);
advance(race, 2.9, {throttle: true});
assert.equal(race.phase, 'countdown');
assert.equal(race.player.distance, 0);
assert.equal(race.opponent.distance, 0);
advance(race, .1, {throttle: true});
assert.equal(race.phase, 'racing');
advance(race, 4, {throttle: true});
assert.ok(race.player.speed > 25 && race.player.speed <= 34);
assert.ok(race.player.distance > 0 && race.opponent.distance > 0);
const cruising = race.player.speed;
advance(race, .5, {});
assert.ok(race.player.speed < cruising && race.player.speed > 0, 'coasting is gradual');
advance(race, 2, {throttle: true, brake: true});
near(race.player.speed, 0, .01);
advance(race, 2, {throttle: true, left: true, right: true});
near(race.player.headingOffset, 0, .001);
const lane = race.player.lateral;
advance(race, .8, {throttle: true, right: true});
assert.ok(race.player.lateral > lane && race.player.headingOffset > 0);
advance(race, 2, {throttle: true});
assert.ok(Math.abs(race.player.headingOffset) < .001, 'heading returns to the road tangent');
advance(race, 3, {throttle: true, right: true});
assert.ok(race.player.lateral > ROAD_HALF_WIDTH);
assert.ok(race.player.speed <= 10, 'off-road speed is bounded');
advance(race, 20, {throttle: true, right: true});
assert.equal(race.player.lateral, 18, 'car remains on the outer shoulder');
near(race.player.headingOffset, 0);
assert.deepEqual(createRace(), initial, 'reset recreates the same pose and countdown');

const laps = createRace();
advance(laps, 3, {throttle: true});
let last = laps.player.distance, previousLaps = 0, lapChanges = 0;
for (let i = 0; i < 8000; i++) {
  stepRace(laps, .1, {throttle: true});
  assert.ok(laps.player.distance >= last);
  const completed = Math.floor(laps.player.distance / TRACK_LENGTH);
  assert.ok(completed === previousLaps || completed === previousLaps + 1);
  lapChanges += completed - previousLaps;
  previousLaps = completed;
  last = laps.player.distance;
}
assert.ok(lapChanges >= 2, 'continuous racing crosses multiple laps');

for (const join of [0, 240, 500, 740, 1000]) {
  const before = trackPose(join - .0001), after = trackPose(join + .0001);
  assert.ok(Math.hypot(before.x - after.x, before.z - after.z) < .001, `track joins at ${join}`);
  const a = trackPose(join - .01), b = trackPose(join), c = trackPose(join + .01);
  const tangentBefore = Math.atan2(b.x - a.x, -(b.z - a.z));
  const tangentAfter = Math.atan2(c.x - b.x, -(c.z - b.z));
  assert.ok(Math.abs(Math.atan2(Math.sin(tangentAfter - tangentBefore), Math.cos(tangentAfter - tangentBefore))) < .001);
}
for (const lateral of [-18, -5, 0, 5, 18]) for (const distance of [-1000, 0, 253, 731, 1000, 2000]) {
  const pose = trackPose(distance, lateral);
  assert.ok(Number.isFinite(pose.x) && Number.isFinite(pose.z) && Number.isFinite(pose.heading));
}
assert.deepEqual(trackPose(0), trackPose(1000));
console.log('PASS racing countdown, motion, braking, steering, shoulder, laps and track joins');
