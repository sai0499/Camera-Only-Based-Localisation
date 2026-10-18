'use strict';
const $ = id => document.getElementById(id);
const S = {frames: [], index: 0, run: null, payload: null, geometry: [], token: '', view: '2d', playing: false, config: {}, plotReady: false, chartReady: false, busy: false};
const ink = '#262b27', red = '#b73826', paper = '#f1efe7', grid = '#dedfd4';
const fmt = (v, digits=2) => v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toFixed(digits);
const text = (id, value) => { $(id).textContent = value; };
const seconds = ns => Number(BigInt(ns) - BigInt(S.payload?.origin_ns || '0')) / 1e9;
const selected = () => S.frames[S.index];
const clean = value => String(value || 'unavailable').replaceAll('_', ' ');
function notice(message) { text('notice', message); $('notice').hidden = !message; }
async function api(path, data) {
  const options = data === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Replay-Token': S.token}, body: JSON.stringify(data)};
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Local request failed');
  return result;
}
function togglePlay(value) {
  S.playing = value && S.frames.length > 1;
  text('play', S.playing ? 'Pause' : 'Play');
  $('play').setAttribute('aria-label', S.playing ? 'Pause replay' : 'Play replay');
  if (S.playing) {
    if (S.index === S.frames.length - 1) choose(0);
    $('latest').checked = false;
    S.playStart = performance.now(); S.playSource = seconds(selected().t_ns);
    requestAnimationFrame(playLoop);
  }
}
let lastPaint = 0;
function playLoop(now) {
  if (!S.playing) return;
  const target = S.playSource + (now - S.playStart) / 1000 * Number($('speed').value);
  if (now - lastPaint > 80) {
    let next = S.index;
    while (next + 1 < S.frames.length && seconds(S.frames[next + 1].t_ns) <= target) next++;
    if (next !== S.index) choose(next);
    if (next === S.frames.length - 1) togglePlay(false);
    lastPaint = now;
  }
  if (S.playing) requestAnimationFrame(playLoop);
}
function choose(index, manual=false) {
  if (!S.frames.length) return;
  S.index = Math.max(0, Math.min(S.frames.length - 1, index));
  if (manual) { togglePlay(false); $('latest').checked = false; }
  $('scrubber').value = S.index;
  const row = selected(), pose = row.pose || {};
  text('selected-time', fmt(seconds(row.t_ns))); text('frame-number', `OUTPUT ${S.index + 1} / ${S.frames.length}`);
  text('absolute-stamp', `EXPOSURE ${row.t_ns} ns`); text('group-id', `GROUP ${row.group_id ?? '—'}`);
  text('pose-x', fmt(row.x)); text('pose-y', fmt(row.y)); text('pose-yaw', fmt(row.yaw * 180 / Math.PI));
  text('tracking-state', clean(row.status)); $('tracking-state').className = 'tracking-state ' + row.status;
  text('status-reason', clean(row.reason)); text('evidence-source', row.evidence?.map(clean).join(' / ') || 'No evidence record');
  text('inliers', pose.inlier_count == null ? '—' : `${pose.inlier_count} / ${fmt(pose.inlier_fraction * 100, 0)}%`);
  text('reprojection', pose.median_error_px == null ? '—' : `${fmt(pose.median_error_px)} / ${fmt(pose.p90_error_px)} px`);
  text('coverage', pose.occupied_cells == null ? '—' : `${pose.occupied_cells} / 16 cells`);
  text('position-error', row.error_m == null ? 'Unavailable' : `${fmt(row.error_m)} m`);
  text('processing', row.processing_ms == null ? 'Unavailable' : `${fmt(row.processing_ms, 1)} ms`);
  const cameras = $('cameras'); cameras.replaceChildren();
  for (const camera of row.images || []) { const line = document.createElement('div'); line.textContent = `${camera.camera_id.toUpperCase()} / ${camera.exposure_ns} ns`; cameras.append(line); }
  if (!cameras.children.length) cameras.textContent = 'Camera timestamps unavailable';
  if (row.missing?.length) { const line = document.createElement('div'); line.textContent = `MISSING / ${row.missing.join(', ')}`; cameras.append(line); }
  scheduleImage();
  const reject = $('rejections'); reject.replaceChildren();
  for (const r of row.rejected || []) { const line = document.createElement('div'); line.textContent = `${r.camera_id || ''} / ${clean(r.reason)}`; reject.append(line); }
  if (!reject.children.length) reject.textContent = 'No pose rejection recorded for this output.';
  text('height-detail', row.z == null ? 'Auxiliary height unavailable at this output.' : `Height ${fmt(row.z)} m · ${clean(row.height_source)} · age ${row.height_source_ns ? fmt((Number(BigInt(row.t_ns) - BigInt(row.height_source_ns))) / 1e9, 2) + ' s' : 'unavailable'}`);
  updateMap();
  if (S.chartReady) Plotly.relayout('timeline', {'shapes[0].x0': seconds(row.t_ns), 'shapes[0].x1': seconds(row.t_ns)}).catch(error => notice(error.message));
}
const axis = title => ({title: {text: title, font: {size: 10}}, gridcolor: grid, zerolinecolor: grid, tickfont: {size: 10}, showline: false});
const config = {displayModeBar: false, responsive: true, scrollZoom: true};
function baseLayout() { return {paper_bgcolor: paper, plot_bgcolor: paper, font: {family: 'IBM Plex Sans', color: ink, size: 11}, showlegend: false, margin: {l: 46, r: 15, t: 5, b: 35}, uirevision: S.run}; }
function arrays(rows, key) { return rows.map(r => r[key] ?? null); }
async function drawMap() {
  if (S.view === 'diagnostics') return;
  if(S.scene3d){S.scene3d.destroy();S.scene3d=null;}
  if(S.view==='3d'){
    Plotly.purge('scene');S.plotReady=false;
    S.scene3d=new ApexScene3D($('scene'),S.geometry);S.scene3d.update(S.frames,S.index);
    text('scene-caption','REAL XYZ · ORTHOGRAPHIC · 1× VERTICAL SCALE');
    text('geometry-note','Drag to orbit. Height gaps remain gaps; no measured GT height is available.');
    return;
  }
  const rows = S.frames, data = [];
  for (const border of S.geometry) {
    const coords = border.xyz;
    data.push({type:'scatter',mode:'lines',name:border.name,x:coords.map(p=>p[0]),y:coords.map(p=>p[1]),line:{color:'#a8ad9d',width:1},hoverinfo:'skip'});
  }
  S.trace = {estimate: data.length};
  const trail = rows.slice(0, S.index + 1);
  data.push({type:'scatter',mode:'lines',x:arrays(trail,'x'),y:arrays(trail,'y'),line:{color:red,width:1.8},name:'Estimate',hoverinfo:'skip'});
  S.trace.lost = data.length;
  const lost=trail.filter(r=>r.status==='lost');
  data.push({type:'scatter',mode:'markers',x:arrays(lost,'x'),y:arrays(lost,'y'),marker:{color:ink,size:5,symbol:'cross'},name:'Lost output',hoverinfo:'skip'});
  S.trace.gt = data.length;
  data.push({type:'scatter',mode:'lines',x:arrays(trail,'gt_x'),y:arrays(trail,'gt_y'),line:{color:ink,width:1.3,dash:'dash'},name:'Ground truth',hoverinfo:'skip'});
  S.trace.current = data.length;
  data.push({type:'scatter',mode:'markers',x:[],y:[],marker:{color:red,size:12,symbol:'triangle-up'},name:'Selected estimate',hovertemplate:'x=%{x:.2f} m<br>y=%{y:.2f} m<extra>Estimate</extra>'});
  S.trace.truth = data.length;
  data.push({type:'scatter',mode:'markers',x:[],y:[],marker:{color:ink,size:7},name:'Selected ground truth',hoverinfo:'skip'});
  S.trace.connector = data.length;
  data.push({type:'scatter',mode:'lines',x:[],y:[],line:{color:ink,width:1,dash:'dot'},hoverinfo:'skip'});
  const layout = baseLayout();
  layout.xaxis=axis('Map x [m]');layout.yaxis={...axis('Map y [m]'),scaleanchor:'x',scaleratio:1};
  text('scene-caption','MAP FRAME · METRES · EQUAL SCALE');
  text('geometry-note','Recorded outputs only. No alignment or smoothing.');
  try { await Plotly.react('scene',data,layout,config); S.plotReady=true; updateMap(); }
  catch (error) { S.plotReady=false; notice(`Scene unavailable: ${error.message}`); }
}
function updateMap() {
  if(S.view==='diagnostics')return;
  if(S.scene3d){S.scene3d.update(S.frames,S.index);return;}
  if (!S.plotReady || S.view==='diagnostics' || !selected()) return;
  const r=selected(),trail=S.frames.slice(0,S.index+1);
  const lost=trail.filter(row=>row.status==='lost');
  const updates=[
    [S.trace.estimate,{x:[arrays(trail,'x')],y:[arrays(trail,'y')]}],
    [S.trace.lost,{x:[arrays(lost,'x')],y:[arrays(lost,'y')]}],
    [S.trace.gt,{x:[arrays(trail,'gt_x')],y:[arrays(trail,'gt_y')]}],
    [S.trace.current,{x:[[r.x]],y:[[r.y]],'marker.angle':90-r.yaw*180/Math.PI}],
    [S.trace.truth,{x:[r.gt_x!=null?[r.gt_x]:[]],y:[r.gt_y!=null?[r.gt_y]:[]]}],
    [S.trace.connector,{x:[r.gt_x!=null?[r.x,r.gt_x]:[]],y:[r.gt_y!=null?[r.y,r.gt_y]:[]]}]
  ];
  for (const [trace, update] of updates) Plotly.restyle('scene',update,[trace]).catch(error=>notice(error.message));
  if ($('follow').checked) Plotly.relayout('scene',{'xaxis.range':[r.x-85,r.x+85],'yaxis.range':[r.y-85,r.y+85]}).catch(error=>notice(error.message));
}
async function drawChart() {
  const key=$('metric').value, title=$('metric').selectedOptions[0].textContent, times=S.frames.map(r=>seconds(r.t_ns)), values=S.frames.map(r=>r[key]??null);
  const lost=S.frames.filter(r=>r.status==='lost');
  const layout={...baseLayout(),margin:{l:49,r:18,t:4,b:28},xaxis:{...axis('Time since initial pose [s]'),range:[0,times.at(-1)||1]},yaxis:axis(title.match(/\[.*\]/)?.[0]||''),shapes:[{type:'line',xref:'x',yref:'paper',x0:seconds(selected()?.t_ns||S.payload.origin_ns),x1:seconds(selected()?.t_ns||S.payload.origin_ns),y0:0,y1:1,line:{color:ink,width:1,dash:'dot'}}]};
  const data=[{type:'scatter',mode:'lines',x:times,y:values,line:{color:red,width:1.2},name:title,connectgaps:false,hovertemplate:'%{x:.2f} s / %{y:.2f}<extra></extra>'},{type:'scatter',mode:'markers',x:lost.map(r=>seconds(r.t_ns)),y:lost.map(r=>r[key]??null),marker:{symbol:'x',color:ink,size:4},name:'Lost output',hovertemplate:'Lost / %{x:.2f} s<extra></extra>'}];
  await Plotly.react('timeline',data,layout,config); S.chartReady=true;
  if (!S.chartBound) { $('timeline').on('plotly_click',event=>{ const target=event.points[0].x; let nearest=0; for(let i=1;i<S.frames.length;i++) if(Math.abs(seconds(S.frames[i].t_ns)-target)<Math.abs(seconds(S.frames[nearest].t_ns)-target)) nearest=i; choose(nearest,true); }); S.chartBound=true; }
  text('chart-note',values.some(v=>v!=null)?'All outputs retained · crosses = Lost · click to seek':'This metric is unavailable without a recorded timing or attached reference');
}
let imageTimer=null,imageSeq=0;
function scheduleImage(delay=S.playing?400:150){clearTimeout(imageTimer);imageTimer=setTimeout(loadImage,delay);}
function loadImage(){
  const row=selected(),img=$('camera-image');if(!row)return;
  const none=message=>{img.removeAttribute('src');text('camera-caption',message);};
  if(window.APEX_REPLAY)return none('Recorded images need the local workspace server.');
  if(S.payload.state!=='completed')return none('Recorded image appears when the run completes, so inference is not contended.');
  const want=$('inspect-camera').value,have=(row.images||[]).map(x=>x.camera_id);
  const camera=have.includes(want)?want:have.includes(row.pose?.camera_id)?row.pose.camera_id:have[0];
  if(!camera)return none('No camera image recorded at this output.');
  const seq=++imageSeq,image=row.images.find(x=>x.camera_id===camera);
  img.onload=()=>{if(seq===imageSeq)text('camera-caption',`${camera.toUpperCase()} · raw recorded image · exposure ${image.exposure_ns} ns${camera!==want?` · ${want.toUpperCase()} not recorded here`:''}`);};
  img.onerror=()=>{if(seq===imageSeq)none(`${camera.toUpperCase()} image unavailable at this output.`);};
  img.src=`/api/image?id=${S.run}&index=${S.index}&camera=${camera}`;
}
function events() {
  const container=$('events');container.replaceChildren();
  for(const event of S.payload.events||[]) {
    const button=document.createElement('button');button.className='event-row';
    button.disabled=event.group_id==null;
    for(const value of [`G ${event.group_id??'—'}`,clean(event.kind),clean(event.reason)]) {const span=document.createElement('span');span.textContent=value;button.append(span);}
    button.onclick=()=>{const i=S.frames.findIndex(r=>r.group_id===event.group_id);if(i>=0)choose(i,true);};container.append(button);
  }
  if(!container.children.length)container.textContent='No recovery, tracker or processing-error events recorded.';
}
async function acceptPayload(payload, reset=false) {
  // Ground truth attached at completion: reload once so every output carries its reference.
  if(!reset&&payload.comparison&&!S.payload?.comparison)return loadRun(S.run);
  if(reset&&payload.comparison&&$('metric').value==='processing_ms')$('metric').value='error_m';
  if(reset) {
    S.frames=[];S.index=0;S.plotReady=false;S.chartReady=false;
    for(const id of ['pose-x','pose-y','pose-yaw','inliers','reprojection','coverage','evidence-source','cameras','processing'])text(id,'—');
    text('group-id','GROUP —');text('tracking-state','Awaiting output');$('tracking-state').className='tracking-state';
    text('status-reason','Waiting for the first completed camera group.');text('position-error','Unavailable');
    text('selected-time','0.00');text('frame-number','OUTPUT —');text('absolute-stamp','EXPOSURE — ns');
    $('camera-image').removeAttribute('src');text('camera-caption','Waiting for an output');
  }
  S.payload=payload;
  for(const row of payload.frames) S.frames[row.index]=row;
  if(reset&&payload.state==='completed')S.index=Math.max(0,S.frames.length-1);
  for(const id of ['play','previous','next','scrubber','next-loss'])$(id).disabled=!S.frames.length;
  text('process-state',clean(payload.state));text('output-count',payload.count.toLocaleString());
  const option=$('run-select').selectedOptions[0];
  if(option?.textContent.endsWith(' / running')&&payload.state!=='running')option.textContent=option.textContent.replace(' / running',` / ${payload.state} / ${payload.count}`);
  text('run-name',payload.meta.query_bag || 'Initialising recording');
  text('duration',S.frames.length?`${fmt(seconds(S.frames.at(-1).t_ns),1)} s processed`:'Waiting for first camera group');
  text('end-time',S.frames.length?fmt(seconds(S.frames.at(-1).t_ns)):'0.00');
  const comparison=payload.comparison;
  text('rmse',comparison?`${fmt(comparison.rmse_m)} m`:'Not evaluated');
  text('reference-state',comparison?`Offline / ${comparison.matched.toLocaleString()} matched outputs`:'No ground truth attached');
  $('gt-legend').hidden=!comparison||S.view==='3d';
  const evaluation=payload.evaluation;
  if(evaluation&&!comparison)text('reference-state',['pending','evaluating'].includes(evaluation.state)?(payload.state==='running'?'Ground truth will be compared when localization finishes.':'Comparing embedded ground truth…'):evaluation.reason||'Reference unavailable');
  $('download-score').hidden=evaluation?.state!=='completed';
  $('download-score').href=`/api/download?id=${S.run}&file=validation/score.txt`;
  $('compare').disabled=!!window.APEX_REPLAY||payload.state!=='completed'||!!evaluation;
  $('new-run').disabled=!!window.APEX_REPLAY||payload.state==='running';
  const mode=!payload.meta.inference_mode?'INITIALIZING':payload.meta.inference_mode==='vehicle_motion_assisted'?'MOTION-ASSISTED':'CAMERA ONLY';
  text('mode-note',`${mode} / ${payload.state==='running'?'PASSIVE RUN MONITOR':'SAVED OUTPUT REVIEW'}`);
  $('scrubber').max=Math.max(0,S.frames.length-1);
  $('download-csv').href=window.APEX_REPLAY?'submission.csv':`/api/download?id=${S.run}&file=submission.csv`;
  if(payload.failure)notice(`${payload.failure.exception}: ${payload.failure.detail}`);
  if($('latest').checked)S.index=Math.max(0,S.frames.length-1);
  if(reset)await drawMap();
  await drawChart();events();choose(Math.min(S.index,Math.max(0,S.frames.length-1)));
}
async function loadRun(id) {
  togglePlay(false);S.run=id;notice('');$('run-select').value=id;text('process-state','Loading replay');
  const payload=await api(`/api/run?id=${id}`);
  if(S.run===id)await acceptPayload(payload,true);
}
async function poll() {
  if(S.run&&!S.busy&&(S.payload?.state==='running'||['pending','evaluating'].includes(S.payload?.evaluation?.state))) {
    const run=S.run;
    S.busy=true;try{const result=await api(`/api/run?id=${run}&cursor=${S.frames.length}`);if(S.run===run)await acceptPayload(result);}catch(error){notice(error.message);}finally{S.busy=false;}
  }
  setTimeout(poll,1000);
}
function requestSettings() {
  const initial=JSON.parse($('initial-pose').value);
  if(typeof initial.timestamp_ns==='number'&&!Number.isSafeInteger(initial.timestamp_ns))throw new Error('Put timestamp_ns in double quotes to preserve exact nanoseconds.');
  return {bag:$('bag').value.trim(),artifacts:$('artifacts').value.trim(),mode:$('run-mode').value,initial_pose:$('initial-pose').value.trim(),retrieval:$('retrieval').checked,vehicle_motion:$('vehicle-motion').value,start_ns:$('start-ns').value.trim(),end_ns:$('end-ns').value.trim()};
}
function uploadRecording(file) {
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();xhr.open('POST','/api/upload');
    xhr.setRequestHeader('X-Replay-Token',S.token);xhr.setRequestHeader('Content-Type','application/octet-stream');
    xhr.setRequestHeader('X-File-Name',encodeURIComponent(file.name));
    $('upload-progress').hidden=false;
    xhr.upload.onprogress=event=>{if(event.lengthComputable){const percent=Math.round(100*event.loaded/event.total);$('upload-progress').value=percent;text('validation-message',`Uploading recording / ${percent}%`);}};
    xhr.onload=()=>{try{const result=JSON.parse(xhr.responseText);if(xhr.status!==200)throw new Error(result.error||'Upload failed');resolve(result);}catch(error){reject(error);}};
    xhr.onerror=()=>reject(new Error('Recording upload failed. Check that the local server is running.'));
    xhr.onabort=()=>reject(new Error('Recording upload was interrupted.'));
    xhr.send(file);
  });
}
function initControls() {
  $('run-select').onchange=()=>loadRun($('run-select').value).catch(error=>notice(error.message));
  for(const button of document.querySelectorAll('[data-view]')) button.onclick=async()=>{
    S.view=button.dataset.view;S.plotReady=false;
    for(const selector of ['.scene-legend','.scene-foot','.scene-actions'])document.querySelector(selector).hidden=S.view==='diagnostics';
    $('follow').disabled=S.view!=='2d';
    $('gt-legend').hidden=!S.payload?.comparison||S.view==='3d';
    for(const tab of document.querySelectorAll('[data-view]')){tab.classList.toggle('active',tab===button);tab.setAttribute('aria-pressed',String(tab===button));}
    $('scene').hidden=S.view==='diagnostics';$('evidence-log').hidden=S.view!=='diagnostics';
    if(S.view!=='diagnostics')await drawMap();
  };
  $('play').onclick=()=>togglePlay(!S.playing);$('previous').onclick=()=>choose(S.index-1,true);$('next').onclick=()=>choose(S.index+1,true);
  $('scrubber').oninput=()=>choose(Number($('scrubber').value),true);
  $('speed').onchange=()=>{if(S.playing)togglePlay(true);};
  $('metric').onchange=()=>drawChart().catch(error=>notice(error.message));
  $('latest').onchange=()=>{if($('latest').checked){togglePlay(false);choose(S.frames.length-1);}};
  $('next-loss').onclick=()=>{const next=S.frames.findIndex((r,i)=>i>S.index&&r.status==='lost'&&S.frames[i-1]?.status!=='lost');if(next>=0)choose(next,true);else notice('No later loss onset in this run.');};
  $('fit-map').onclick=()=>{$('follow').checked=false;if(S.scene3d){S.scene3d.fit();return;}Plotly.relayout('scene',{'xaxis.autorange':true,'yaxis.autorange':true}).catch(error=>notice(error.message));};
  $('follow').onchange=()=>updateMap();
  $('save-image').onclick=async()=>{
    const button=$('save-image');button.disabled=true;
    try{
      const image=S.scene3d?S.scene3d.canvas.toDataURL('image/png'):await Plotly.toImage('scene',{format:'png',width:1600,height:1000});
      if(window.APEX_REPLAY){const link=document.createElement('a');link.href=image;link.download='apex-scene.png';link.click();}
      else{const result=await api('/api/snapshot',{id:S.run,image});notice(`Scene saved: ${result.path}`);const link=document.createElement('a');link.href=result.url;link.target='_blank';link.rel='noopener';link.textContent=' Open snapshot ↗';$('notice').append(link);}
    }catch(error){notice(error.message);}finally{button.disabled=false;}
  };
  $('new-run').onclick=()=>$('setup-dialog').showModal();
  for(const close of document.querySelectorAll('[data-close]'))close.onclick=()=>$(close.dataset.close).close();
  $('validate-run').onclick=async()=>{const button=$('validate-run');button.disabled=true;text('validation-message','Reading calibration and checking frozen-map compatibility…');try{const result=await api('/api/validate',requestSettings());$('validation-message').classList.remove('error');text('validation-message',`Compatible / ${result.map_id} / cameras ${result.cameras.join(', ')}. Full asset verification runs at startup.`);}catch(error){$('validation-message').classList.add('error');text('validation-message',error.message);}finally{button.disabled=false;}};
  $('mcap-file').onchange=()=>{S.uploaded=null;$('validation-message').classList.remove('error');const f=$('mcap-file').files[0];text('file-summary',f?`${f.name} / ${(f.size/1024**2).toFixed(1)} MB`:'Select a recording to begin.');};
  $('setup-form').onsubmit=async event=>{
    event.preventDefault();$('start-run').disabled=true;$('validation-message').classList.remove('error');
    try{
      let result;
      if($('advanced-settings').open){text('validation-message','Validating manual inputs…');result=await api('/api/start',requestSettings());}
      else{
        const file=$('mcap-file').files[0];if(!file)throw new Error('Choose an MCAP file first.');
        const uploaded=S.uploaded||(S.uploaded=await uploadRecording(file));
        $('upload-progress').hidden=true;text('validation-message','Detecting initial pose and motion, then starting localization…');
        result=await api('/api/start-auto',{bag:uploaded.bag});
      }
      const option=document.createElement('option');option.value=result.id;option.textContent=result.name+' / running';$('run-select').prepend(option);$('latest').checked=true;$('setup-dialog').close();await loadRun(result.id);
    }catch(error){$('validation-message').classList.add('error');text('validation-message',error.message);}
    finally{$('start-run').disabled=false;$('upload-progress').hidden=true;}
  };
  $('compare').onclick=()=>$('gt-dialog').showModal();
  $('gt-form').onsubmit=async event=>{event.preventDefault();$('attach-gt').disabled=true;text('gt-message','Verifying and matching reference timestamps…');try{await acceptPayload(await api('/api/compare',{id:S.run,path:$('gt-path').value.trim()}),true);$('gt-dialog').close();$('metric').value='error_m';await drawMap();await drawChart();}catch(error){text('gt-message',error.message);}finally{$('attach-gt').disabled=false;}};
  $('inspect-camera').onchange=()=>scheduleImage(0);
  for(const button of document.querySelectorAll('[data-browse]'))button.onclick=async()=>{
    button.disabled=true;
    try{const result=await api('/api/browse',{kind:button.dataset.browse});const target=$(button.dataset.target);
      if(result.path){target.value=button.dataset.browse==='pose'?result.content:result.path;target.dispatchEvent(new Event('change'));}}
    catch(error){notice(error.message);}finally{button.disabled=false;}
  };
  window.addEventListener('keydown',event=>{if(/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)||document.querySelector('dialog[open]'))return;if(event.code==='Space'){event.preventDefault();togglePlay(!S.playing);}if(event.key==='ArrowLeft'){event.preventDefault();choose(S.index-1,true);}if(event.key==='ArrowRight'){event.preventDefault();choose(S.index+1,true);}});
  window.addEventListener('resize',()=>{if(S.scene3d)S.scene3d.resize();else if(S.plotReady)Plotly.Plots.resize('scene');if(S.chartReady)Plotly.Plots.resize('timeline');});
}
async function start() {
  initControls();
  if(window.APEX_REPLAY){
    const saved=window.APEX_REPLAY;S.run='offline';S.geometry=saved.geometry;
    const option=document.createElement('option');option.value='offline';option.textContent=saved.name+' / offline bundle';$('run-select').append(option);$('run-select').disabled=true;
    await acceptPayload(saved.payload,true);text('mode-note','OFFLINE SAVED REPLAY');return;
  }
  const [configuration,geometry]=await Promise.all([api('/api/config'),api('/api/scene')]);
  S.config=configuration;S.token=configuration.token;S.geometry=geometry;
  for(const run of configuration.runs){const option=document.createElement('option');option.value=run.id;option.textContent=`${run.name} / ${run.split || 'input'} / ${run.groups??'in progress'}`;$('run-select').append(option);}
  for(const name of configuration.bags){const option=document.createElement('option');option.value=name;$('bag-options').append(option);}
  for(const name of configuration.artifacts){const option=document.createElement('option');option.value=name;$('artifact-options').append(option);}
  $('bag').value=configuration.bags[0]||'';$('artifacts').value=configuration.default_artifacts;
  const initial=configuration.initial_pose;delete initial.record;
  $('initial-pose').value=JSON.stringify(initial,(_,v)=>typeof v==='bigint'?v.toString():v,2);$('gt-path').value=configuration.default_gt;
  $('run-mode').value=initial.timestamp_ns?'tune':'input';
  const defaultBag=$('bag').value;
  $('bag').addEventListener('change',()=>{if($('bag').value!==defaultBag){$('run-mode').value='input';$('initial-pose').value='';}});
  const run=configuration.runs.find(r=>r.running)||configuration.runs.find(r=>r.groups&&r.split==='tune')||configuration.runs.find(r=>r.groups);
  if(run)await loadRun(run.id);else{$('setup-dialog').showModal();}
  poll();
}
start().catch(error=>notice(`Workspace could not load: ${error.message}`));
