import {SCENES} from './game.js';

// Scene capabilities belong here; the session transport remains shared.
export const sourceScenes = {
  boxing: {...SCENES.boxing, label: '拳击 · 双人对战', kind: 'synthetic', resetLabel: '重开对战'},
  racing: {...SCENES.racing, label: '赛车 · 环道竞速', kind: 'synthetic', resetLabel: '重开比赛'},
  torcs: {label: 'TORCS 赛车 · 多风格 Proxy', kind: 'torcs', configKey: 'torcs_source',
    style: 'racing-proxy/v1', reference: 'rgb', resetLabel: '重开比赛',
    help: '点击画面后用 WASD／方向键驾驶，空格手刹。会话前可选择 Proxy 风格和自动驾驶。'},
  ballpoint: {label: 'Ballpoint Breach · 游戏原画 RGB', kind: 'ballpoint', configKey: 'ballpoint_source',
    style: 'ballpoint-rgb/v1', reference: 'rgb', resetLabel: '重开射击游戏',
    help: '点击画面开始鼠标控制，WASD 移动，左键射击，右键瞄准，空格跳跃，R 换弹，Q 钩索，1–5 切换武器。Esc 释放鼠标。'},
  native_game: {label: '本机游戏 · 原生来源', kind: 'native', configKey: 'native_game_source',
    protocol: 'native_game_commands/v1', style: 'native-measured-proxy-jpeg/v1', resetLabel: '重置本机游戏',
    help: '点击画面，WASD 移动，拖动视角，V 切换视角，F 上下车，R 重置。慢源会重复采样帧。'},
  street: {...SCENES.street, label: '街景 · 漫游', kind: 'synthetic'},
};

export function sceneAvailable(id, config) {
  const scene = sourceScenes[id];
  return !!(config && scene && (!scene.configKey || config[scene.configKey]));
}
