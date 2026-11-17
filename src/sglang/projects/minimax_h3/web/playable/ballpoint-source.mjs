import {rgbaToRgb} from './stream.mjs';

const KEYS=new Set(['KeyW','KeyA','KeyS','KeyD','ShiftLeft','ShiftRight','Space','KeyR','KeyQ','Digit1','Digit2','Digit3','Digit4','Digit5']);

export function createBallpointSource(canvas,{width,height,url,onInput=()=>{},onReady=()=>{},onError=()=>{},onPointerStatus=()=>{}}={}){
  canvas.width=width;canvas.height=height;canvas.tabIndex=0;
  canvas.setAttribute('aria-label','Ballpoint Breach，点击画面后使用鼠标瞄准，WASD 移动');
  const context=canvas.getContext('2d',{willReadFrequently:true});
  const iframe=document.createElement('iframe');iframe.hidden=true;iframe.tabIndex=-1;iframe.title='Ballpoint Breach 游戏源';
  let game,disposed=false,failed=false,suspended=false,seconds=0,settle,mouseFallback=false,pendingLock=false;
  const ready=new Promise((resolve,reject)=>{settle={resolve,reject};});
  const timeout=setTimeout(()=>loadFailed(Error('射击游戏加载超时，请先构建 streaming 版本')),30000);
  ready.catch(error=>{if(!disposed)onError(error);});
  function draw(){if(game&&!disposed){context.drawImage(game.canvas,0,0,width,height);game.overlay(context,width,height);}}
  function loadFailed(error){if(disposed||failed)return;failed=true;clearTimeout(timeout);settle.reject(error);}
  function loaded(){
    if(disposed||game||failed)return;
    try{
      const source=iframe.contentWindow.ballpoint;if(!source)return;
      if(!source.canvas||['step','reset','suspend','overlay','state'].some(name=>typeof source[name]!=='function')){
        throw Error('Ballpoint streaming 构建缺少 canvas / step / reset / suspend / overlay / state 接口');
      }
      game=source;draw();clearTimeout(timeout);settle.resolve();onReady();
    }catch(error){game=null;loadFailed(error);}
  }
  iframe.onload=()=>{
    try{iframe.contentWindow.addEventListener('ballpoint-ready',loaded,{once:true});loaded();}
    catch(error){loadFailed(error);}
  };
  iframe.onerror=()=>loadFailed(Error('射击游戏页面加载失败'));
  iframe.src=new URL(url,location.href).href;document.body.append(iframe);
  function emit(type,init){
    if(!game||disposed||suspended)return;
    const target=iframe.contentWindow;
    const EventClass=type.startsWith('key')?target.KeyboardEvent:type==='wheel'?target.WheelEvent:target.MouseEvent;
    target.dispatchEvent(new EventClass(type,{...init,bubbles:true,cancelable:true}));
  }
  function key(event){
    if(event.code==='Escape'){
      pendingLock=false;mouseFallback=false;
      if(document.pointerLockElement===canvas)document.exitPointerLock();
      release();onPointerStatus('鼠标已释放，点击画面重新开始操作。');return;
    }
    if(!KEYS.has(event.code))return;
    event.preventDefault();onInput({type:event.type,code:event.code,...(suspended?{disposition:'ignored_during_startup_pause'}:{})});
    emit(event.type,{code:event.code,key:event.key,repeat:event.repeat});
  }
  function mouse(event){
    if(event.type==='mousedown'&&event.target===canvas){
      canvas.focus();
      if(document.pointerLockElement!==canvas&&!mouseFallback){
        if(pendingLock)return;
        pendingLock=true;
        try{Promise.resolve(canvas.requestPointerLock()).catch(pointerError);}catch{pointerError();}
        return;
      }
    }
    const locked=document.pointerLockElement===canvas;
    if(!locked&&(!mouseFallback||document.activeElement!==canvas||event.type!=='mouseup'&&event.target!==canvas))return;
    if(event.type==='contextmenu'){event.preventDefault();return;}
    if(event.type==='wheel')event.preventDefault();
    onInput({type:event.type,button:event.button,movementX:event.movementX,movementY:event.movementY});
    emit(event.type,{button:event.button,buttons:event.buttons,movementX:event.movementX,movementY:event.movementY,deltaY:event.deltaY});
  }
  const release=()=>{if(game){game.suspend(true);if(!suspended)game.suspend(false);}};
  function pointerChange(){
    if(document.pointerLockElement===canvas){pendingLock=false;mouseFallback=false;onPointerStatus('鼠标已锁定，按 Esc 释放。');}
    else{pendingLock=false;mouseFallback=false;release();onPointerStatus('鼠标已释放，点击画面重新开始操作。');}
  }
  function pointerError(){
    if(!pendingLock||disposed)return;
    pendingLock=false;mouseFallback=true;release();
    onPointerStatus('鼠标锁定不可用，已启用画布内控制；移出画布停止转向，按 Esc 退出。');
  }
  function leave(){if(mouseFallback)release();}
  function blur(){
    pendingLock=false;mouseFallback=false;release();
    onPointerStatus('鼠标已释放，点击画面重新开始操作。');
  }
  for(const name of ['keydown','keyup'])canvas.addEventListener(name,key);
  canvas.addEventListener('mousedown',mouse);canvas.addEventListener('contextmenu',mouse);
  for(const name of ['mouseup','mousemove','wheel'])window.addEventListener(name,mouse,{passive:false});
  canvas.addEventListener('blur',blur);window.addEventListener('blur',blur);
  canvas.addEventListener('mouseleave',leave);
  document.addEventListener('pointerlockchange',pointerChange);document.addEventListener('pointerlockerror',pointerError);
  function render(delta=0){
    if(!game||disposed)return;
    if(!suspended){const dt=Math.max(0,Math.min(Number.isFinite(delta)?delta:0,.05));seconds+=dt;game.step(dt);}
    draw();
  }
  function setControlsSuspended(value){suspended=!!value;game?.suspend(suspended);}
  function reset(){if(game&&!disposed){seconds=0;game.reset();game.suspend(suspended);draw();}}
  async function prepareRun(){await ready;if(disposed)throw Error('射击游戏来源已释放');reset();setControlsSuspended(true);}
  function capture(){if(!game||disposed)throw Error('射击游戏尚未就绪');return rgbaToRgb(context.getImageData(0,0,width,height).data);}
  function referenceCanvas(outputWidth,outputHeight){
    if(!game||disposed)throw Error('射击游戏尚未就绪');
    const output=document.createElement('canvas');output.width=outputWidth;output.height=outputHeight;
    const ctx=output.getContext('2d');ctx.drawImage(game.canvas,0,0,outputWidth,outputHeight);
    game.overlay(ctx,outputWidth,outputHeight);return output;
  }
  function getSourceInfo(){return game?{proxy:{format:'ballpoint-rgb/v1',encoding:'rgb',trainingAligned:false},
    source:{width:game.canvas.width,height:game.canvas.height},capture:{width,height,resize:'full frame; no crop'},
    reference:'same-state game RGB with recording HUD overlay; full frame; no crop'}:null;}
  function getSimulationState(){return game?{...game.state(),time:seconds,scene:'ballpoint',state:suspended?'held':'live',
    controls_suspended:suspended,source:getSourceInfo()}:{state:'loading'};}
  function dispose(){
    disposed=true;clearTimeout(timeout);release();settle.reject(Error('射击游戏来源已释放'));
    for(const name of ['keydown','keyup'])canvas.removeEventListener(name,key);
    canvas.removeEventListener('mousedown',mouse);canvas.removeEventListener('contextmenu',mouse);
    for(const name of ['mouseup','mousemove','wheel'])window.removeEventListener(name,mouse);
    canvas.removeEventListener('blur',blur);window.removeEventListener('blur',blur);
    canvas.removeEventListener('mouseleave',leave);
    document.removeEventListener('pointerlockchange',pointerChange);document.removeEventListener('pointerlockerror',pointerError);
    if(document.pointerLockElement===canvas)document.exitPointerLock();iframe.remove();
  }
  return {width,height,render,reset,prepareRun,capture,referenceCanvas,getSourceInfo,getSimulationState,
    setControlsSuspended,stop:()=>setControlsSuspended(false),dispose};
}
