//server that connects music and emotion 
const EMOTION = '/emotion-api';     
const $ = id => document.getElementById(id), sleep = ms => new Promise(r => setTimeout(r, ms));
let run = 0, running = false, ctx, nextTime = 0, cur = {v: 0, a: 0}, stream, abort;

// cuts old sesssion so start nvr leaves an old session playing alongside the new one
const live = me => me === run && running;
function show(v, a, name){ $('vv').textContent = v.toFixed(2); $('av').textContent = a.toFixed(2); if (name) $('en').textContent = name; }


// webcam, emotion backend, one frame at a time
async function camLoop(me){          
  const video = $('cam'), c = document.createElement('canvas'); c.width = 320; c.height = 240;
  while (live(me)){
    if ($('manual').checked){ cur = {v: +$('sv').value, a: +$('sa').value}; show(cur.v, cur.a, 'sliders'); await sleep(300); continue; }
    try {
      c.getContext('2d').drawImage(video, 0, 0, 320, 240);
      const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.8));
      const fd = new FormData(); fd.append('file', blob, 'frame.jpg');
      const e = await (await fetch(EMOTION + '/frame', {method: 'POST', body: fd, signal: abort.signal})).json();
      if (!live(me)) return;
      if (e.detected){ cur = {v: e.valence, a: e.arousal}; show(cur.v, cur.a, e.emotion_name); }
      else $('en').textContent = 'no face';
      if (e.batch) $('batch').textContent = `last 10 s: ${e.batch.emotion_name} ` +
        `(valence ${e.batch.valence.toFixed(2)}, arousal ${e.batch.arousal.toFixed(2)})`;
    } catch (err) { if (!live(me)) return; $('en').textContent = 'emotion backend offline'; await sleep(1000); }
    await sleep(250);
  }
}

// chunks play back to back and  each is requested as late as possible so it use newest mood to play
async function musicLoop(me){        
  const sid = 'run' + me + '-' + Math.random().toString(36).slice(2);   
  let first = true, makeTime = 3;
  $('log').textContent = 'making the first music (about 5 s)...';
  while (live(me)){
    if (nextTime - ctx.currentTime > makeTime + 0.8){ await sleep(50); continue; }
    const asked = performance.now();
    let r, buf;
    try {
      r = await fetch(`/live/next?valence=${cur.v}&arousal=${cur.a}&session=${sid}${first ? '&reset=true' : ''}`,
                      {signal: abort.signal});
      if (!live(me)) return;
      if (!r.ok){ $('log').textContent = 'music error: ' + await r.text(); await sleep(1000); continue; }
      buf = await ctx.decodeAudioData(await r.arrayBuffer());
    } catch (err) { if (!live(me)) return; $('log').textContent = 'music backend offline'; await sleep(1000); continue; }
    if (!live(me)) return;
    if (!first) makeTime = 0.7 * makeTime + 0.3 * (performance.now() - asked) / 1000;
    const info = JSON.parse(r.headers.get('X-Music-Info'));
    const src = ctx.createBufferSource(); src.buffer = buf; src.connect(ctx.destination);
    const late = !first && nextTime < ctx.currentTime;
    const t = Math.max(nextTime, ctx.currentTime + 0.05); src.start(t); nextTime = t + buf.duration;
    first = false;
    $('log').textContent = `music mood: valence ${info.music_valence}, arousal ${info.music_arousal}\n` +
      `made ${info.seconds_audio} s of music in ${info.seconds_to_make} s` + (late ? '  (fell behind: small gap)' : '');
  }
}

// silence everything now, queued chunks get deleted, camera no longer take input
function stop(){
  running = false; run++;
  abort && abort.abort();
  ctx && ctx.close(); ctx = null;
  stream && stream.getTracks().forEach(t => t.stop()); stream = null; $('cam').srcObject = null;
  $('btn').textContent = 'Start'; $('log').textContent = 'stopped';
}

async function start(){
  const me = ++run; running = true; $('btn').textContent = 'Stop';
  abort = new AbortController();
  ctx = new AudioContext(); nextTime = 0;
  // some browsers start audio paused until told to play
  await ctx.resume();               
  if (!$('manual').checked){
    try { stream = await navigator.mediaDevices.getUserMedia({video: true}); $('cam').srcObject = stream; }
    catch (e) { $('manual').checked = true; $('log').textContent = 'no camera access, using the sliders'; }
    if (!live(me)) { stream && stream.getTracks().forEach(t => t.stop()); return; }
  }
  camLoop(me); musicLoop(me);
}

function toggle(){ running ? stop() : start(); }
