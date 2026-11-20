import assert from 'node:assert/strict';
import {CAPSULES, createMatch, fighterPose, startAttack, stepMatch} from '../web/playable/boxing-logic.mjs';

const match = createMatch();
stepMatch(match, .1);
assert.equal(match.phase, 'ready', 'the opponent waits for the first player action');
for (let i = 0; i < 200; i++) stepMatch(match, .1, {left: true, down: true});
assert.equal(match.fighters[0].x, -2.15);
assert.equal(match.fighters[0].z, 1.45);

const crowded = createMatch();
crowded.fighters[0].x = 1.45;
crowded.fighters[0].z = 0;
crowded.fighters[1].x = 2.05;
crowded.fighters[1].z = 0;
for (let i = 0; i < 180; i++) {
  stepMatch(crowded, 1 / 60, {right: true});
  assert.ok(crowded.fighters[0].x < crowded.fighters[1].x);
  assert.ok(Math.hypot(crowded.fighters[0].x - crowded.fighters[1].x,
    crowded.fighters[0].z - crowded.fighters[1].z) >= .55);
  assert.ok(crowded.fighters.every(fighter => Math.abs(fighter.x) <= 2.15 && Math.abs(fighter.z) <= 1.45));
}
const blockedX = crowded.fighters[0].x;
stepMatch(crowded, 1 / 60, {left: true});
assert.ok(crowded.fighters[0].x < blockedX, 'the player can retreat from the opponent');

const fight = createMatch();
const [player, opponent] = fight.fighters;
player.x = -.5; opponent.x = .5;
player.z = opponent.z = 0;
opponent.cooldown = 100;
assert.equal(startAttack(player, 'jab'), true);
stepMatch(fight, .1);
stepMatch(fight, .1);
assert.equal(opponent.hp, 91, 'one jab lands once when in range');
assert.equal(startAttack(player, 'heavy'), false, 'a punch cannot interrupt the active punch');
stepMatch(fight, .1);
stepMatch(fight, .1);
assert.equal(opponent.hp, 91, 'the hit does not repeat during recovery');
assert.equal(startAttack(player, 'jab'), false, 'cooldown blocks another punch');
for (let i = 0; i < 3; i++) stepMatch(fight, .1);
assert.equal(startAttack(player, 'heavy'), true);
for (let i = 0; i < 4; i++) stepMatch(fight, .1);
assert.equal(opponent.hp, 73);

for (let i = 0; i < 100 && fight.phase !== 'result'; i++) {
  stepMatch(fight, .1);
  if (!player.attack && player.cooldown === 0) startAttack(player, 'heavy');
}
assert.equal(fight.phase, 'result');
assert.equal(fight.winner, 'player');
assert.equal(opponent.hp, 0);
assert.deepEqual(createMatch().fighters.map(f => f.hp), [100, 100], 'reset returns both fighters to full health');

assert.equal(CAPSULES.length, 15);
assert.equal(new Set(CAPSULES.map(([name]) => name)).size, 15);
for (const fighter of fight.fighters) {
  const pose = fighterPose(fighter, fight.fighters[1 - fight.fighters.indexOf(fighter)]);
  for (const [name, a, b, radius] of CAPSULES) {
    assert.ok(radius > 0 && radius < .2, name);
    assert.ok([...(pose[a] ?? []), ...(pose[b] ?? [])].length === 6, name);
    assert.ok([...pose[a], ...pose[b]].every(Number.isFinite), name);
  }
}
console.log('PASS boxing bounds, fighter separation, ready state, hit timing, cooldown, KO, reset, and 15 finite capsules');
