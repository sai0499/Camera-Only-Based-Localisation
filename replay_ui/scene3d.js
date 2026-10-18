'use strict';
// Orthographic XYZ inspection keeps the passive viewer off the inference GPU.
window.ApexScene3D = class {
  constructor(host,geometry) {
    this.host=host;this.geometry=geometry;this.rows=[];this.index=0;
    this.yaw=-0.5;this.pitch=0.65;this.zoom=1;this.drag=null;
    this.canvas=document.createElement('canvas');this.canvas.setAttribute('aria-label','Rotatable orthographic 3D track geometry. Drag to orbit; use mouse wheel to zoom.');
    this.canvas.tabIndex=0;this.canvas.style.cssText='width:100%;height:100%;display:block;touch-action:none;cursor:grab';
    host.replaceChildren(this.canvas);
    const points=geometry.flatMap(border=>border.xyz);
    this.centre=[0,1,2].map(k=>(Math.min(...points.map(p=>p[k]))+Math.max(...points.map(p=>p[k])))/2);
    this.radius=Math.max(...points.map(p=>Math.hypot(...p.map((v,k)=>v-this.centre[k]))),1);
    this.canvas.onpointerdown=e=>{this.drag=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);this.canvas.style.cursor='grabbing';};
    this.canvas.onpointermove=e=>{if(!this.drag)return;this.yaw+=(e.clientX-this.drag[0])*0.007;this.pitch=Math.max(-1.4,Math.min(1.4,this.pitch+(e.clientY-this.drag[1])*0.007));this.drag=[e.clientX,e.clientY];this.draw();};
    this.canvas.onpointerup=()=>{this.drag=null;this.canvas.style.cursor='grab';};
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=Math.max(0.4,Math.min(12,this.zoom*Math.exp(-e.deltaY*0.001)));this.draw();},{passive:false});
    this.canvas.onkeydown=e=>{if(e.key.startsWith('Arrow')){e.preventDefault();e.stopPropagation();this.yaw+=(e.key==='ArrowRight'?0.1:e.key==='ArrowLeft'?-0.1:0);this.pitch=Math.max(-1.4,Math.min(1.4,this.pitch+(e.key==='ArrowDown'?0.1:e.key==='ArrowUp'?-0.1:0)));this.draw();}};
    this.resize();
  }
  resize(){const rect=this.host.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);this.width=rect.width;this.height=rect.height;this.canvas.width=Math.round(rect.width*dpr);this.canvas.height=Math.round(rect.height*dpr);this.ctx=this.canvas.getContext('2d');this.ctx.setTransform(dpr,0,0,dpr,0,0);this.draw();}
  project(point){const [x,y,z]=point.map((v,k)=>v-this.centre[k]),c=Math.cos(this.yaw),s=Math.sin(this.yaw),scale=Math.min(this.width,this.height)*0.43/this.radius*this.zoom;const u=c*x-s*y,v=s*x+c*y;return[this.width/2+u*scale,this.height/2-(v*Math.sin(this.pitch)+z*Math.cos(this.pitch))*scale];}
  path(points,colour,width){const ctx=this.ctx;ctx.beginPath();let open=false;for(const point of points){if(!point||point.some(v=>v==null)){open=false;continue;}const [x,y]=this.project(point);if(open)ctx.lineTo(x,y);else ctx.moveTo(x,y);open=true;}ctx.strokeStyle=colour;ctx.lineWidth=width;ctx.stroke();}
  update(rows,index){this.rows=rows;this.index=index;this.draw();}
  fit(){this.zoom=1;this.yaw=-0.5;this.pitch=0.65;this.draw();}
  draw(){
    const ctx=this.ctx;if(!ctx||!this.width)return;
    ctx.clearRect(0,0,this.width,this.height);ctx.fillStyle='#f1efe7';ctx.fillRect(0,0,this.width,this.height);
    for(const border of this.geometry)this.path(border.xyz,'#949b89',1);
    this.path(this.rows.slice(0,this.index+1).map(r=>r.z==null?null:[r.x,r.y,r.z]),'#b73826',2);
    ctx.strokeStyle='#262b27';ctx.lineWidth=1;
    for(const r of this.rows.slice(0,this.index+1)){if(r.status!=='lost'||r.z==null)continue;const [x,y]=this.project([r.x,r.y,r.z]);ctx.beginPath();ctx.moveTo(x-2,y);ctx.lineTo(x+2,y);ctx.moveTo(x,y-2);ctx.lineTo(x,y+2);ctx.stroke();}
    const row=this.rows[this.index];
    if(row?.z!=null){const [x,y]=this.project([row.x,row.y,row.z]);ctx.fillStyle='#b73826';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();}
    ctx.font='11px "IBM Plex Sans"';ctx.fillStyle='#65685f';ctx.fillText('ORTHOGRAPHIC / DRAG TO ORBIT / SCROLL TO ZOOM',18,24);
    ctx.fillText('Map XYZ · metres · vertical scale 1×',18,this.height-15);
    if(row&&row.z==null)ctx.fillText('Selected pose has no valid auxiliary height',18,43);
    // Axis triad uses the same rotation and equal-unit scale, independent of map translation.
    const origin=[this.width-65,this.height-55],colours=['#262b27','#65685f','#b73826'];
    for(let k=0;k<3;k++){const tip=[...this.centre];tip[k]+=this.radius*0.16/this.zoom;const q=this.project(tip);const end=[origin[0]+q[0]-this.width/2,origin[1]+q[1]-this.height/2];ctx.strokeStyle=colours[k];ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...end);ctx.stroke();ctx.fillStyle=colours[k];ctx.fillText(['X','Y','Z'][k],end[0]+4,end[1]-2);}
  }
  destroy(){this.host.replaceChildren();}
};
