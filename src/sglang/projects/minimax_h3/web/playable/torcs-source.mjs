import {rgbaToRgb} from './stream.mjs';

const KEYS=new Set(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space','KeyR']);

export function createTorcsSource(canvas,{width,height,url,onInput=()=>{},onReady=()=>{},onError=()=>{},canReset=()=>true}={}){
  canvas.width=width;canvas.height=height;
  canvas.tabIndex=0;canvas.setAttribute('aria-label','TORCS 赛车 Proxy，WASD 驾驶，空格手刹');
  const context=canvas.getContext('2d',{willReadFrequently:true});
  context.imageSmoothingEnabled=false;
  const iframe=document.createElement('iframe');
  iframe.hidden=true;iframe.tabIndex=-1;iframe.title='TORCS 赛车渲染源';
  let racing,disposed=false,suspended=false,accumulator=0,settle;
  const ready=new Promise((resolve,reject)=>{settle={resolve,reject};});
  const timeout=setTimeout(()=>settle.reject(Error('赛车加载超时，请先构建 streaming 版本')),30000);
  ready.catch(error=>{if(!disposed)onError(error);});
  function loaded(){
    if(disposed||racing)return;
    racing=iframe.contentWindow.racing;
    if(!racing)return;
    clearTimeout(timeout);racing.semantic.resize(width,height);
    racing.render();draw();settle.resolve();onReady();
  }
  iframe.onload=()=>{
    iframe.contentWindow.addEventListener('torcs-ready',loaded,{once:true});
    loaded();
  };
  iframe.onerror=()=>{clearTimeout(timeout);settle.reject(Error('赛车页面加载失败'));};
  iframe.src=new URL(url,location.href).href;document.body.append(iframe);

  function draw(){if(racing&&!disposed)context.drawImage(racing.semantic.canvas,0,0,width,height);}
  function release(){racing?.controls.clear();}
  function key(event){
    if(!KEYS.has(event.code))return;
    event.preventDefault();
    if(!racing||disposed)return;
    const ignored=suspended;
    onInput({type:event.type,code:event.code,...(ignored?{disposition:'ignored_during_startup_pause'}:{})});
    if(ignored)return;
    if(event.code==='KeyR'){
      if(event.type==='keydown'&&!event.repeat&&canReset())reset();
    }else if(event.type==='keydown')racing.controls.add(event.code);
    else racing.controls.delete(event.code);
  }
  const focus=()=>canvas.focus();
  for(const name of ['keydown','keyup'])canvas.addEventListener(name,key);
  canvas.addEventListener('blur',release);canvas.addEventListener('pointerdown',focus);
  window.addEventListener('blur',release);

  function render(delta=0){
    if(!racing||disposed)return;
    if(!suspended){
      accumulator+=Math.max(0,Math.min(Number.isFinite(delta)?delta:0,.1));
      while(accumulator>=1/120){racing.step(1/120);accumulator-=1/120;}
    }
    racing.render();draw();
  }
  function setControlsSuspended(value){
    if(value&&!suspended)onInput({disposition:'released_on_startup_pause',type:'release'});
    suspended=!!value;release();accumulator=0;
  }
  function reset(){
    if(!racing||disposed)return;
    release();accumulator=0;racing.reset();draw();
  }
  async function prepareRun(){
    await ready;
    if(disposed)throw Error('赛车来源已释放');
    reset();setControlsSuspended(true);
  }
  function capture(){
    if(!racing||disposed)throw Error('赛车 Proxy 尚未就绪');
    return rgbaToRgb(context.getImageData(0,0,width,height).data);
  }
  function referenceCanvas(outputWidth,outputHeight){
    if(!racing||disposed)throw Error('赛车 RGB 尚未就绪');
    const reference=document.createElement('canvas');reference.width=outputWidth;reference.height=outputHeight;
    reference.getContext('2d').drawImage(racing.rgbCanvas,0,0,outputWidth,outputHeight);
    return reference;
  }
  function getSourceInfo(){
    return racing?{proxy:racing.semantic.descriptor(),autopilot:racing.state().auto,
      source:racing.semantic.mode==='rgb'?{width:racing.rgbCanvas.width,height:racing.rgbCanvas.height}:{width:racing.semantic.canvas.width,height:racing.semantic.canvas.height},
      capture:{width,height,resize:racing.semantic.mode==='rgb'?'game RGB scaled to proxy geometry; full frame; no crop':'none; proxy rendered at artifact geometry; full frame; no crop'},reference:'same-state RGB; full frame; no crop'}:null;
  }
  function getSimulationState(){return racing?{...racing.state(),scene:'torcs',state:suspended?'held':'live',controls_suspended:suspended,source:getSourceInfo()}: {state:'loading'};}
  function dispose(){
    disposed=true;clearTimeout(timeout);release();settle.reject(Error('赛车来源已释放'));
    for(const name of ['keydown','keyup'])canvas.removeEventListener(name,key);
    canvas.removeEventListener('blur',release);canvas.removeEventListener('pointerdown',focus);
    window.removeEventListener('blur',release);iframe.contentWindow?.removeEventListener('torcs-ready',loaded);iframe.remove();
  }
  return {width,height,render,reset,prepareRun,capture,referenceCanvas,getSourceInfo,getSimulationState,
    setControlsSuspended,stop:()=>setControlsSuspended(false),dispose,
    modes:()=>racing?.proxyModes??[],
    setProxyMode:id=>{if(!canReset())throw Error('会话期间不能切换 Proxy');racing.setProxyMode(id);draw();},
    setAutopilot:value=>{if(!canReset())throw Error('会话期间不能切换自动驾驶');racing.setAuto(value);},
  };
}
