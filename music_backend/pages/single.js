const v=document.getElementById('v'),a=document.getElementById('a'),info=document.getElementById('info');
// same as from face library based on Russell circumplex's model
const NAMES=[[16,'pleased'],[34,'happy'],[62.5,'delighted'],[78.5,'excited'],[93,'astonished'],[104,'aroused'],
  [115,'tensed'],[126,'alarmed'],[137,'afraid'],[148,'annoyed'],[159,'distressed'],[170,'frustrated'],[181,'miserable'],
  [192,'sad'],[203,'gloomy'],[215,'depressed'],[230,'bored'],[245,'droopy'],[260,'tired'],[280,'sleepy'],
  [300,'calm'],[320,'serene'],[340,'content'],[354,'satisfied'],[361,'pleased']];
function emoteName(valence,arousal){
  const strength=Math.hypot(valence,arousal);
  if(strength<0.1) return 'Neutral';
  const level=strength<0.325?'Slightly':strength<0.55?'Moderately':strength<0.775?'Very':'Extremely';
  const angle=(Math.atan2(arousal,valence)*180/Math.PI+360)%360;
  return level+' '+NAMES.find(n=>angle<n[0])[1];
}
function update(){
  vv.textContent=v.value; av.textContent=a.value;
  document.getElementById('emote').textContent=emoteName(+v.value,+a.value);
}
v.oninput=update; a.oninput=update;
async function go(url){
  info.textContent='generating (takes a few seconds on CPU)...';
  const r=await fetch(url);
  if(!r.ok){info.textContent='error: '+await r.text();return;}
  info.textContent=r.headers.get('X-Music-Info');
  player.src=URL.createObjectURL(await r.blob()); player.play();
}
