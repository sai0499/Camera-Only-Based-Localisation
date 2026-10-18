const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

let now = 0, nextTimer = 1;
const timers = new Map(), requests = [];
const image = {removeAttribute() {}, set src(url) { requests.push({url, time: now}); this.onload?.(); }};
const elements = {'camera-image': image, 'inspect-camera': {value: 'fr'}, 'camera-caption': {textContent: ''}};
const context = vm.createContext({window: {}, document: {getElementById: id => elements[id]},
  setTimeout(callback, delay) { const id = nextTimer++; timers.set(id, {callback, due: now + delay}); return id; },
  clearTimeout(id) { timers.delete(id); }});
const source = fs.readFileSync('replay_ui/app.js', 'utf8');
assert.match(source, /\nstart\(\)\.catch/);
vm.runInContext(source.slice(0, source.lastIndexOf('\nstart().catch')), context);
vm.runInContext(`S.run='replay'; S.payload={state:'completed'}; S.playing=true;
  S.frames=Array.from({length:30},(_,i)=>({images:[{camera_id:'fr',exposure_ns:String(i)},{camera_id:'fl',exposure_ns:String(i)}]}));`, context);
function advance(target) {
  for (;;) {
    const next = [...timers].sort((a,b) => a[1].due-b[1].due)[0];
    if (!next || next[1].due > target) break;
    now = next[1].due; timers.delete(next[0]); next[1].callback();
  }
  now = target;
}
// Continuous 10 Hz playback must load images before playback ends.
for (let i = 0; i < 30; i++) {
  advance(i * 100);
  vm.runInContext(`S.index=${i}; scheduleImage();`, context);
}
assert.ok(requests.length >= 10, 'Playback starved camera image updates');
const indexes = requests.map(r => Number(new URL(r.url, 'http://localhost').searchParams.get('index')));
assert.ok(indexes[0] < 5 && indexes.at(-1) > 20);
assert.ok(indexes.every((index,i) => !i || index > indexes[i-1]));
vm.runInContext('S.playing=false; S.index=29; scheduleImage();', context);
advance(now);
assert.match(requests.at(-1).url, /index=29&camera=fr$/);
elements['inspect-camera'].value = 'fl';
vm.runInContext('scheduleImage(0);', context); advance(now);
assert.match(requests.at(-1).url, /index=29&camera=fl$/);
console.log('Playback camera previews update continuously, settle on the final frame and follow camera selection.');
