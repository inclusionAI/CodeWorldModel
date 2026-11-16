export const CAPSULES = [
  ['spine', 'pelvis', 'neck', .048], ['neck', 'neck', 'head', .044], ['head', 'head', 'head', .115],
  ...[-1, 1].flatMap((side, index) => {
    const s = index ? 'right' : 'left';
    return [
      [`${s}_shoulder`, 'neck', `${s}_upperArm`, .044],
      [`${s}_upper_arm`, `${s}_upperArm`, `${s}_forearm`, .038],
      [`${s}_forearm`, `${s}_forearm`, `${s}_hand`, .038],
      [`${s}_hip`, 'pelvis', `${s}_thigh`, .044],
      [`${s}_thigh`, `${s}_thigh`, `${s}_calf`, .042],
      [`${s}_calf`, `${s}_calf`, `${s}_foot`, .042],
    ];
  }),
];

const ATTACK = {jab: {duration: .34, impact: .13, damage: 9, reach: 1.34, cooldown: .25},
  heavy: {duration: .58, impact: .28, damage: 18, reach: 1.48, cooldown: .55}};
const clamp = (value, limit) => Math.max(-limit, Math.min(limit, value));

export function createMatch() {
  return {phase: 'ready', winner: null, fighters: [
    {x: -1.15, z: .2, hp: 100, attack: null, cooldown: 0, step: 0},
    {x: 1.15, z: -.2, hp: 100, attack: null, cooldown: .45, step: 0},
  ]};
}

export function startAttack(fighter, kind) {
  if (!ATTACK[kind] || fighter.attack || fighter.cooldown > 0 || fighter.hp <= 0) return false;
  fighter.attack = {kind, time: 0, hit: false};
  return true;
}

export function stepMatch(match, dt, controls = {}) {
  dt = Number.isFinite(dt) ? Math.max(0, Math.min(dt, .1)) : 0;
  const [player, opponent] = match.fighters;
  const mx = Number(!!controls.right) - Number(!!controls.left);
  const mz = Number(!!controls.down) - Number(!!controls.up);
  if (match.phase === 'ready' && (mx || mz || player.attack)) match.phase = 'playing';
  if (match.phase !== 'playing') return match;
  const scale = dt * 1.9 / (Math.hypot(mx, mz) || 1);
  const nextX = clamp(player.x + mx * scale, 2.15);
  const nextZ = clamp(player.z + mz * scale, 1.45);
  if (Math.hypot(nextX - opponent.x, nextZ - opponent.z) >= .55) {
    player.x = nextX;
    player.z = nextZ;
  }
  if (mx || mz) player.step += dt * 12;
  const dx = player.x - opponent.x, dz = player.z - opponent.z;
  const distance = Math.hypot(dx, dz);
  if (distance > 1.08) {
    const speed = dt * 1.15;
    opponent.x = clamp(opponent.x + dx / distance * speed, 2.15);
    opponent.z = clamp(opponent.z + dz / distance * speed, 1.45);
    opponent.step += dt * 9;
  } else if (opponent.cooldown <= 0 && !opponent.attack) startAttack(opponent, 'jab');
  for (const [actor, target] of [[player, opponent], [opponent, player]]) {
    actor.cooldown = Math.max(0, actor.cooldown - dt);
    const attack = actor.attack;
    if (!attack) continue;
    attack.time += dt;
    const spec = ATTACK[attack.kind];
    if (!attack.hit && attack.time >= spec.impact) {
      attack.hit = true;
      if (Math.hypot(actor.x - target.x, actor.z - target.z) <= spec.reach) {
        target.hp = Math.max(0, target.hp - spec.damage);
        if (!target.hp) {
          match.phase = 'result';
          match.winner = actor === player ? 'player' : 'opponent';
          break;
        }
      }
    }
    if (attack.time >= spec.duration) {
      actor.attack = null;
      actor.cooldown = actor === opponent ? 1.15 : spec.cooldown;
    }
  }
  return match;
}

export function fighterPose(fighter, target) {
  const stride = Math.sin(fighter.step) * .13;
  const forward = fighter.attack ? Math.sin(Math.PI * Math.min(1, fighter.attack.time / ATTACK[fighter.attack.kind].duration)) : 0;
  const reach = fighter.attack?.kind === 'heavy' ? .73 : .49;
  const points = {pelvis: [0, .88, 0], neck: [0, 1.53, -.04], head: [0, 1.72, -.04]};
  for (const [side, sign] of [['left', -1], ['right', 1]]) {
    const punching = fighter.attack && (fighter.attack.kind === 'heavy' ? sign === 1 : sign === -1);
    const extension = punching ? forward * reach : 0;
    points[`${side}_upperArm`] = [sign * .22, 1.47, -.03];
    points[`${side}_forearm`] = [sign * (.34 - extension * .16), 1.22 + extension * .15, -.13 - extension * .55];
    points[`${side}_hand`] = [sign * (.28 - extension * .25), 1.38 + extension * .14, -.32 - extension];
    points[`${side}_thigh`] = [sign * .13, .78, sign * stride * .35];
    points[`${side}_calf`] = [sign * .15, .42, sign * stride];
    points[`${side}_foot`] = [sign * .16, .12, -.09 + sign * stride * 1.7];
  }
  const yaw = Math.atan2(target.x - fighter.x, -(target.z - fighter.z));
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return Object.fromEntries(Object.entries(points).map(([name, [x, y, z]]) =>
    [name, [fighter.x + c * x - s * z, y, fighter.z + s * x + c * z]]));
}
