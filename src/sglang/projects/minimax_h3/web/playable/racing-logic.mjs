export const TRACK_LENGTH = 1000;
export const ROAD_HALF_WIDTH = 5;
const STRAIGHT = 240;
const RADIUS = (TRACK_LENGTH - 2 * STRAIGHT) / (2 * Math.PI);
const SHOULDER_LIMIT = 18;

export function trackPose(distance, lateral = 0) {
  const d = ((distance % TRACK_LENGTH) + TRACK_LENGTH) % TRACK_LENGTH;
  let x, z, heading;
  if (d < STRAIGHT) {
    x = 0; z = -d; heading = 0;
  } else if (d < TRACK_LENGTH / 2) {
    const angle = (d - STRAIGHT) / RADIUS;
    x = RADIUS * (1 - Math.cos(angle)); z = -STRAIGHT - RADIUS * Math.sin(angle); heading = angle;
  } else if (d < TRACK_LENGTH / 2 + STRAIGHT) {
    x = 2 * RADIUS; z = -STRAIGHT + d - TRACK_LENGTH / 2; heading = Math.PI;
  } else {
    const angle = (d - TRACK_LENGTH / 2 - STRAIGHT) / RADIUS;
    x = RADIUS * (1 + Math.cos(angle)); z = RADIUS * Math.sin(angle); heading = Math.PI + angle;
  }
  return {x: x + lateral * Math.cos(heading), z: z + lateral * Math.sin(heading), heading};
}

export function createRace() {
  return {phase: 'ready', countdown: 3,
    player: {distance: 0, lateral: -1.8, speed: 0, headingOffset: 0},
    opponent: {distance: 0, lateral: 1.8, speed: 0, headingOffset: 0}};
}

export function stepRace(race, dt, input = {}) {
  dt = Number.isFinite(dt) ? Math.max(0, Math.min(dt, .1)) : 0;
  if (race.phase === 'ready') {
    if (!input.throttle) return race;
    race.phase = 'countdown';
  }
  if (race.phase === 'countdown') {
    const waiting = Math.min(dt, race.countdown);
    race.countdown -= waiting;
    dt -= waiting;
    if (race.countdown > 1e-9) return race;
    race.countdown = 0;
    race.phase = 'racing';
  }
  while (dt > 1e-9) {
    const step = Math.min(dt, 1 / 120);
    dt -= step;
    const car = race.player;
    const offroad = Math.abs(car.lateral) > ROAD_HALF_WIDTH;
    const limit = offroad ? 10 : 34;
    const acceleration = input.brake ? -26 : input.throttle ? 15 : -3;
    car.speed = Math.max(0, Math.min(limit, car.speed + (acceleration - (offroad ? 9 : 0)) * step));
    const steer = Number(!!input.right) - Number(!!input.left);
    car.headingOffset += (steer * .65 - car.headingOffset) * (1 - Math.exp(-7 * step));
    car.distance += car.speed * Math.cos(car.headingOffset) * step;
    car.lateral = Math.max(-SHOULDER_LIMIT, Math.min(SHOULDER_LIMIT,
      car.lateral + car.speed * Math.sin(car.headingOffset) * step));
    if (Math.abs(car.lateral) === SHOULDER_LIMIT && Math.sign(car.headingOffset) === Math.sign(car.lateral)) car.headingOffset = 0;
    const rival = race.opponent;
    rival.speed = Math.min(27, rival.speed + 10 * step);
    rival.distance += rival.speed * step;
  }
  return race;
}
