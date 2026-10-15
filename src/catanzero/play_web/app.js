'use strict';
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const palette={RED:'#c45845',BLUE:'#377ba4',WHITE:'#a9946c',ORANGE:'#d49132'};
const resources={WOOD:{name:'Wood',color:'#71915d'},BRICK:{name:'Brick',color:'#b76d50'},SHEEP:{name:'Sheep',color:'#a0b66d'},WHEAT:{name:'Wheat',color:'#d0ac51'},ORE:{name:'Ore',color:'#84929d'}};
const devNames={KNIGHT:'Knight',YEAR_OF_PLENTY:'Year of Plenty',MONOPOLY:'Monopoly',ROAD_BUILDING:'Road Building',VICTORY_POINT:'Victory Point'};
const purchases=[{type:'BUILD_ROAD',name:'Road',cost:{WOOD:1,BRICK:1}},{type:'BUILD_SETTLEMENT',name:'Settlement',cost:{WOOD:1,BRICK:1,SHEEP:1,WHEAT:1}},{type:'BUILD_CITY',name:'City',cost:{WHEAT:2,ORE:3}},{type:'BUY_DEVELOPMENT_CARD',name:'Dev. card',cost:{SHEEP:1,WHEAT:1,ORE:1}}];
const spatial=new Set(['BUILD_ROAD','BUILD_SETTLEMENT','BUILD_CITY','MOVE_ROBBER']);
let table=null,bots=['402'],busy=false,mode=null,pendingPlacement=null,victimChoices=[],selectedDev=null;
let tradeGive='WOOD',tradeGet='BRICK',botTimer=null,toastTimer=null,retryNeeded=false;
let fitBox={x:-320,y:-290,width:640,height:580},previousHand=null,changes={};
const storageKey=globalThis.catanStorageKey||'catanzero-table';
const resourceName=r=>resources[r]?.name||'Desert';
const own=()=>table?.players.find(p=>p.human);
const legal=type=>table?.legal_actions.filter(a=>a.type===type)||[];
const playerName=color=>{const p=table?.players.find(p=>p.color===color);return p&&!p.human&&p.model==='402'?p.name.split(' · ')[0]:p?.name||color;};
function element(tag,cls,text,parent){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;if(parent)parent.append(e);return e;}
function svg(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);if(parent)parent.append(e);return e;}
function svgText(parent,x,y,text,cls){const e=svg('text',{x,y,class:cls},parent);e.textContent=text;return e;}
function button(parent,text,callback,cls=''){const e=element('button',cls,text,parent);e.onclick=callback;e.disabled=busy;return e;}
function remember(id){try{if(id)localStorage.setItem(storageKey,id);else localStorage.removeItem(storageKey);}catch{}}
function notify(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,5000);}
async function api(path,body){if(globalThis.catanOfflineApi)return globalThis.catanOfflineApi(path,body);const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok){const e=new Error(data.error||'Request failed.');e.status=r.status;throw e;}return data;}
function renderBotCount(){
  $('variant-note').textContent=bots.length===1?'2-player variant on the standard board.':`${bots.length+1} players on the standard board.`;
  document.querySelectorAll('[data-count]').forEach(e=>e.setAttribute('aria-pressed',String(+e.dataset.count===bots.length)));
}
function hex(x,y,r=62){return Array.from({length:6},(_,i)=>{const a=(-90+i*60)*Math.PI/180;return`${x+r*Math.cos(a)},${y+r*Math.sin(a)}`;}).join(' ');}
function portPosition(tile,geometry){const ends=(tile.edge||[]).map(n=>geometry.nodes[n]).filter(Boolean);if(ends.length!==2)return{x:tile.x,y:tile.y};const x=(ends[0][0]+ends[1][0])/2,y=(ends[0][1]+ends[1][1])/2;return{x:x+.45*(tile.x-x),y:y+.45*(tile.y-y)};}
function fitBoard(geometry){
  const points=Object.values(geometry.nodes).map(p=>[...p]);
  for(const t of geometry.tiles){if(t.kind==='port'){const p=portPosition(t,geometry);points.push([p.x-20,p.y-20],[p.x+20,p.y+20]);}else points.push([t.x-56,t.y-64],[t.x+56,t.y+64]);}
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const x=Math.min(...xs)-8,y=Math.min(...ys)-8;
  fitBox={x,y,width:Math.max(...xs)-x+8,height:Math.max(...ys)-y+8};
  $('board').setAttribute('viewBox',`${x} ${y} ${fitBox.width} ${fitBox.height}`);
}
function drawArt(tile,parent){
  const g=svg('g',{transform:`translate(${tile.x},${tile.y-23})`,class:'tile-art'},parent);
  if(tile.resource==='WOOD')for(const[x,y]of[[-19,4],[0,-7],[19,4]]){svg('path',{d:`M${x} ${y-22}l-14 26h28z`,fill:'#35583d'},g);svg('path',{d:`M${x} ${y+2}v10`,stroke:'#35583d','stroke-width':4},g);}
  else if(tile.resource==='ORE'){svg('path',{d:'M-31 14L-12-20L1 2L14-27L34 14Z',fill:'#4d606c'},g);svg('path',{d:'M14-27L4-10L14-15L22-10Z',fill:'#dce1d6'},g);}
  else if(tile.resource==='WHEAT')for(const x of[-15,0,15]){svg('path',{d:`M${x} 13v-35m0 8l-6-5m6 13l6-5m-6 13l-6-5`,stroke:'#886b2d','stroke-width':3,fill:'none'},g);}
  else if(tile.resource==='BRICK')for(const[x,y]of[[-28,1],[-7,1],[14,1],[-17,-12],[4,-12]])svg('rect',{x,y,width:19,height:11,rx:1,fill:'#854b37'},g);
  else if(tile.resource==='SHEEP'){svg('path',{d:'M-12 5v13M8 5v13',stroke:'#60724b','stroke-width':3},g);svg('ellipse',{cx:0,cy:0,rx:23,ry:14,fill:'#f4f1db'},g);svg('ellipse',{cx:22,cy:0,rx:7,ry:9,fill:'#60724b'},g);}
  else svg('path',{d:'M-31 12Q-11-7 4 8Q23-4 32 12Z',fill:'#b2986d'},g);
}
function drawBoard(geometry,state=null){
  for(const id of['tiles','ports','roads','buildings','robber','targets','preview-piece'])$(id).replaceChildren();
  fitBoard(geometry);
  for(const t of geometry.tiles){
    if(t.kind==='port'){
      const p=portPosition(t,geometry);
      for(const n of t.edge||[]){const v=geometry.nodes[n];if(v)svg('line',{x1:p.x,y1:p.y,x2:v[0],y2:v[1],stroke:'#64828c','stroke-width':2},$('ports'));}
      const g=svg('g',{'aria-label':`${t.resource?resourceName(t.resource):'Any resource'} port, ${t.resource?'2:1':'3:1'}`},$('ports'));
      svg('circle',{cx:p.x,cy:p.y,r:17,fill:t.resource?resources[t.resource].color:'#e0d9bf',stroke:'#f2ead0','stroke-width':2},g);
      svgText(g,p.x,p.y+5,t.resource?'2:1':'3:1','port-text');const title=svg('title',{},g);title.textContent=g.getAttribute('aria-label');continue;
    }
    const g=svg('g',{},$('tiles')),r=resources[t.resource];
    svg('polygon',{points:hex(t.x,t.y),fill:r?.color||'#cfbb90',stroke:'#e7dbba','stroke-width':3,'stroke-linejoin':'round'},g);
    drawArt(t,g);
    if(t.number){svg('circle',{cx:t.x,cy:t.y+17,r:21,fill:'#f7efdc',stroke:'#d0c4a2'},g);svgText(g,t.x,t.y+25,t.number,`tile-number ${[6,8].includes(t.number)?'hot':''}`);svgText(g,t.x,t.y+37,'•'.repeat(6-Math.abs(7-t.number)),'pips');}
    svgText(g,t.x,t.y+52,r?.name||'Desert','tile-resource');
  }
  if(!state)return;
  const seen=new Set();
  for(const[edge,color]of state.roads){const id=[...edge].sort((a,b)=>a-b).join('-');if(seen.has(id))continue;seen.add(id);const a=geometry.nodes[edge[0]],b=geometry.nodes[edge[1]],coords={x1:a[0],y1:a[1],x2:b[0],y2:b[1],'stroke-linecap':'round'};svg('line',{...coords,stroke:'#1b303c','stroke-width':15,class:'road-outline'},$('roads'));svg('line',{...coords,stroke:palette[color],'stroke-width':10,class:'road-piece'},$('roads'));svg('line',{...coords,stroke:'#ffffff55','stroke-width':2,'pointer-events':'none'},$('roads'));}
  for(const[n,color,kind]of state.buildings)drawBuilding(geometry.nodes[n],palette[color],kind,$('buildings'));
  const robbed=geometry.tiles.find(t=>JSON.stringify(t.coordinate)===JSON.stringify(state.robber));
  if(robbed){const g=svg('g',{transform:`translate(${robbed.x+34},${robbed.y-2})`},$('robber'));svg('circle',{cx:0,cy:-12,r:7,fill:'#25313a',stroke:'#e1dfcd','stroke-width':1.5},g);svg('path',{d:'M-6-5L-11 17Q0 22 11 17L6-5Z',fill:'#25313a',stroke:'#e1dfcd','stroke-width':1.5},g);}
  drawTargets();
}
function drawBuilding(point,color,kind,parent,preview=false){const g=svg('g',{transform:`translate(${point[0]},${point[1]})`,filter:'url(#piece-shadow)'},parent);svg('path',{d:kind==='CITY'?'M-14 10V-3L-6-12L1-3V-11L9-20L17-11V10Z':'M-11 10V-3L0-15L11-3V10Z',fill:color,stroke:'#fff9e6','stroke-width':2.5,'stroke-linejoin':'round',class:preview?'preview-building':'piece'},g);}
function chooseMode(type){mode=type;pendingPlacement=null;victimChoices=[];selectedDev=null;renderTurn();renderPurchases();renderSpecial();drawTargets();}
function selectBoardAction(index){if(busy||!table?.human_turn)return;pendingPlacement={index,revision:table.revision};victimChoices=[];renderTurn();renderSpecial();drawTargets();}
function bindTarget(target,label,callback){target.setAttribute('role','button');target.setAttribute('tabindex','0');target.setAttribute('aria-label',label);target.onclick=event=>{event?.stopPropagation();if(event&&event.detail!==0&&mode!=='MOVE_ROBBER')selectNearest(event);else callback();};target.onkeydown=event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();callback();}};svg('title',{},target).textContent=label;}
function locationLabel(a){
  if(a.type==='MOVE_ROBBER'){const t=table.geometry.tiles.find(t=>JSON.stringify(t.coordinate)===JSON.stringify(a.value[0]));return`Move robber to ${resourceName(t?.resource)} ${t?.number||''}`;}
  const points=(a.type==='BUILD_ROAD'?a.value:[a.value]).map(n=>table.geometry.nodes[n]);
  const nearby=table.geometry.tiles.filter(t=>t.kind==='land'&&points.some(p=>Math.hypot(t.x-p[0],t.y-p[1])<65));
  return`${purchases.find(p=>p.type===a.type)?.name||'Build'}: ${nearby.map(t=>`${resourceName(t.resource)} ${t.number||''}`).join(', ')}`;
}
function chooseRobberTile(actions){if(actions.length===1)selectBoardAction(actions[0].index);else{victimChoices=actions;pendingPlacement=null;renderTurn();renderSpecial();}}
function drawTargets(){
  $('targets').replaceChildren();$('preview-piece').replaceChildren();if(!table?.human_turn||busy||!spatial.has(mode))return;
  const actions=legal(mode),nodes=table.geometry.nodes;
  if(mode==='MOVE_ROBBER'){
    const handled=new Set();for(const a of actions){const key=JSON.stringify(a.value[0]);if(handled.has(key))continue;handled.add(key);const tile=table.geometry.tiles.find(t=>JSON.stringify(t.coordinate)===key);const choices=actions.filter(b=>JSON.stringify(b.value[0])===key);const g=svg('g',{class:`map-target ${choices.some(b=>pendingPlacement?.index===b.index)?'selected':''}`},$('targets'));svg('polygon',{points:hex(tile.x,tile.y,58),class:'target-tile'},g);bindTarget(g,locationLabel(a),()=>chooseRobberTile(choices));}
  }else for(const a of actions){
    const selected=pendingPlacement?.index===a.index,g=svg('g',{class:`map-target ${selected?'selected':''}`},$('targets'));
    if(mode==='BUILD_ROAD'){const p=nodes[a.value[0]],q=nodes[a.value[1]],coords={x1:p[0],y1:p[1],x2:q[0],y2:q[1]};svg('line',{...coords,class:'target-road-border'},g);svg('line',{...coords,class:'target-road-hit'},g);svg('line',{...coords,class:'target-road'},g);if(selected)svg('line',{...coords,class:'preview-road'},$('preview-piece'));}
    else{const p=nodes[a.value];svg('circle',{cx:p[0],cy:p[1],r:8,class:'target-hit'},g);svg('circle',{cx:p[0],cy:p[1],r:8,class:'target-node'},g);if(selected)drawBuilding(p,'#edb24c',mode==='BUILD_CITY'?'CITY':'SETTLEMENT',$('preview-piece'),true);}
    bindTarget(g,locationLabel(a),()=>selectBoardAction(a.index));
  }
}
function selectNearest(event){
  if(!table?.human_turn||busy||!spatial.has(mode)||mode==='MOVE_ROBBER')return;
  const rect=$('board').getBoundingClientRect(),scale=Math.min(rect.width/fitBox.width,rect.height/fitBox.height);
  const x=(event.clientX-rect.left-(rect.width-fitBox.width*scale)/2)/scale+fitBox.x;
  const y=(event.clientY-rect.top-(rect.height-fitBox.height*scale)/2)/scale+fitBox.y;
  let best=null,distance=Infinity;
  for(const a of legal(mode)){let d;if(mode==='BUILD_ROAD'){const p=table.geometry.nodes[a.value[0]],q=table.geometry.nodes[a.value[1]],dx=q[0]-p[0],dy=q[1]-p[1],t=Math.max(0,Math.min(1,((x-p[0])*dx+(y-p[1])*dy)/(dx*dx+dy*dy)));d=Math.hypot(x-p[0]-t*dx,y-p[1]-t*dy)+.01*Math.hypot(x-(p[0]+q[0])/2,y-(p[1]+q[1])/2);}else{const p=table.geometry.nodes[a.value];d=Math.hypot(x-p[0],y-p[1]);}if(d<distance){distance=d;best=a;}}
  if(best&&distance*scale<=24)selectBoardAction(best.index);
}
function renderPlayers(){
  $('players').replaceChildren();for(const p of table.players){const card=element('div',`player ${table.current===p.color&&!table.winner?'active':''}`,undefined,$('players'));card.style.setProperty('--player',palette[p.color]);const heading=element('div','player-heading',undefined,card);const name=element('strong','',playerName(p.color),heading);name.title=name.textContent;element('b','score',`${p.human?p.actual_vp:p.VICTORY_POINTS}`,heading);const details=element('div','player-details',undefined,card);element('span','',`${p.resource_count} res.`,details);element('span','',`${p.development_count} dev.`,details);if(p.HAS_ROAD||p.HAS_ARMY){const badge=element('span','player-award',[p.HAS_ROAD?'Longest road':'',p.HAS_ARMY?'Largest army':''].filter(Boolean).join(' · '),card);badge.title=badge.textContent;}}
}
function renderHand(){
  const p=own(),now=Date.now();$('hand-count').textContent=`${p.resource_count} cards`;$('hand').replaceChildren();
  for(const[key,r]of Object.entries(resources)){const n=p.resources[key];if(previousHand&&previousHand[key]!==n)changes[key]={delta:n-previousHand[key],until:now+4000};const change=changes[key]?.until>now?changes[key].delta:0;const card=element('div',`resource ${change>0?'gained':change<0?'spent':''}`,undefined,$('hand'));card.style.setProperty('--resource',r.color);card.setAttribute('aria-label',`${n} ${r.name}${change?', changed by '+change:''}`);element('span','resource-name',r.name,card);element('b','',String(n),card);if(change)element('span','resource-change',`${change>0?'+':''}${change}`,card);}
  previousHand={...p.resources};
}
function effectiveCost(item){return table.initial_phase&&['BUILD_ROAD','BUILD_SETTLEMENT'].includes(item.type)||item.type==='BUILD_ROAD'&&table.free_roads_available?{}:item.cost;}
function renderPurchases(){
  $('purchases').replaceChildren();for(const item of purchases){const actions=legal(item.type),cost=effectiveCost(item);const b=button($('purchases'),'',()=>item.type==='BUY_DEVELOPMENT_CARD'?perform(actions[0].index):chooseMode(item.type),`purchase ${mode===item.type?'selected':''}`);b.disabled=busy||!actions.length;element('strong','',item.name,b);const price=element('span','price',undefined,b);for(const[r,n]of Object.entries(cost)){const chip=element('span',own().resources[r]>=n?'':'missing',`${n} ${resources[r].name}`,price);chip.style.setProperty('--resource',resources[r].color);chip.title=`${n} ${resources[r].name}`;chip.setAttribute('aria-label',chip.title);}if(!Object.keys(cost).length)element('span','free','Free',price);b.title=actions.length?item.name:Object.entries(cost).some(([r,n])=>own().resources[r]<n)?'Not enough resources':!table.human_turn?'Wait for your turn':'No legal action available';}
  $('build-note').textContent=table.initial_phase?'Setup: settlement + road':mode&&spatial.has(mode)?'Select a highlighted location.':'Prices shown below each item.';
}
function tradeRatio(key){let ratio=4;for(const t of table.geometry.tiles.filter(t=>t.kind==='port'))if((t.edge||[]).some(n=>table.buildings.some(([node,color])=>node===n&&color===table.human))){if(t.resource===key)ratio=2;else if(!t.resource)ratio=Math.min(ratio,3);}return ratio;}
function tradeAction(){return legal('MARITIME_TRADE').find(a=>{const offered=a.value.slice(0,-1).filter(Boolean);return a.value.at(-1)===tradeGet&&tradeGet!==tradeGive&&offered.length>0&&offered.every(r=>r===tradeGive);});}
function tradeBlockReason(ratio){
  if(table.winner)return'Game over.';
  if(busy)return'Waiting for the current move to finish.';
  if(!table.human_turn)return'Available on your turn.';
  if(table.initial_phase)return'Finish initial placement before trading.';
  if(mode==='DISCARD_RESOURCE'||table.prompt==='DISCARD')return'Discard the required resources before trading.';
  if(mode==='MOVE_ROBBER'||table.prompt==='MOVE_ROBBER')return'Move the robber before trading.';
  if(legal('ROLL').length||!own().HAS_ROLLED)return'Roll the dice before trading.';
  if(table.free_roads_available)return'Place your free roads before trading.';
  if(tradeGive===tradeGet)return'Choose a different resource to receive.';
  if(own().resources[tradeGive]<ratio)return`Need ${ratio} of the same resource: ${resourceName(tradeGive)}. You have ${own().resources[tradeGive]}.`;
  if(!table.resource_bank[tradeGet])return`The bank has no ${resourceName(tradeGet)}. Choose another resource.`;
  return'No legal trade for this selection.';
}
function renderTrade(){
  for(const option of $('trade-give').children)option.textContent=`${resourceName(option.value)} (${own().resources[option.value]})`;
  for(const option of $('trade-get').children){option.disabled=option.value===tradeGive;option.textContent=resourceName(option.value);}
  $('trade-give').value=tradeGive;$('trade-get').value=tradeGet;const a=tradeAction(),ratio=a?a.value.slice(0,-1).filter(Boolean).length:tradeRatio(tradeGive),reason=tradeBlockReason(ratio);
  $('trade-ratio').textContent=`${ratio}:1`;$('trade-submit').textContent=table.human_turn&&!table.initial_phase&&legal('ROLL').length?'Roll first':`Trade ${ratio}:1`;$('trade-submit').disabled=busy||!a;
  $('trade-submit').title=a?`Exchange ${ratio} ${resourceName(tradeGive)} for 1 ${resourceName(tradeGet)}`:reason;
  $('trade-message').textContent=a&&!busy?`Give ${ratio} ${resourceName(tradeGive)} → receive 1 ${resourceName(tradeGet)}`:reason;
}
function renderTurn(){
  if(!table)return;
  if(pendingPlacement?.revision!==table.revision)pendingPlacement=null;
  const pending=pendingPlacement&&table.legal_actions.find(a=>a.index===pendingPlacement.index);
  $('placement-panel').hidden=!pending;$('quick-controls').hidden=!!pending;
  if(pending){$('placement-title').textContent=pending.type==='MOVE_ROBBER'?'Move robber?':`Build ${purchases.find(p=>p.type===pending.type)?.name.toLowerCase()}?`;$('confirm-placement').textContent=pending.type==='MOVE_ROBBER'?'Move':'Build';$('confirm-placement').disabled=busy;}
  $('retry-bot').hidden=!retryNeeded||table.human_turn;
  const primary=$('quick-primary');primary.disabled=busy;$('quick-end').hidden=true;
  if(table.winner){$('quick-title').textContent=table.winner===table.human?'You won':'Game over';$('quick-hint').textContent=`${playerName(table.winner)} wins.`;primary.textContent='New game';primary.onclick=()=>$('new-dialog').showModal();return;}
  if(!table.human_turn){$('quick-title').textContent=playerName(table.current);$('quick-hint').textContent=retryNeeded?'Move failed. Use Retry.':'Playing…';primary.textContent='Waiting';primary.disabled=true;return;}
  const roll=legal('ROLL')[0],end=legal('END_TURN')[0];
  const titles={BUILD_SETTLEMENT:'Place a settlement',BUILD_ROAD:'Place a road',BUILD_CITY:'Upgrade a settlement',MOVE_ROBBER:'Move the robber',DISCARD_RESOURCE:'Discard resources'};
  $('quick-title').textContent=roll?'Your turn':victimChoices.length?'Choose who to steal from':titles[mode]||'Your turn';
  $('quick-hint').textContent=roll?'Roll to collect resources.':spatial.has(mode)?'Select a marked location, then confirm.':mode==='DISCARD_RESOURCE'?'Choose a resource to discard below.':'Build, trade, play a card, or end your turn.';
  primary.textContent=roll?'Roll dice':end?'End turn':'Select on board';primary.disabled=busy||(!roll&&!end);primary.onclick=()=>roll?perform(roll.index):end?perform(end.index):undefined;
}
function renderDevelopment(){
  $('dev-cards').replaceChildren();$('dev-count').textContent=String(own().development_count);
  const typeFor={KNIGHT:'PLAY_KNIGHT_CARD',YEAR_OF_PLENTY:'PLAY_YEAR_OF_PLENTY',MONOPOLY:'PLAY_MONOPOLY',ROAD_BUILDING:'PLAY_ROAD_BUILDING'};
  for(const[key,n]of Object.entries(own().development_cards)){if(!n)continue;const a=legal(typeFor[key]);const b=button($('dev-cards'),`${devNames[key]} ×${n}`,()=>{if(a.length===1)perform(a[0].index);else{selectedDev=typeFor[key];pendingPlacement=null;renderSpecial();}},'dev-card');b.disabled=busy||!a.length;b.title=a.length?'Play card':'Not playable now';}
  if(!own().development_count)element('span','empty-cards','No development cards.',$('dev-cards'));
}
function resourceSelect(parent,id,value){const select=element('select','',undefined,parent);select.id=id;select.setAttribute('aria-label',id==='dev-first'?'First resource':'Second resource');for(const[key,r]of Object.entries(resources)){const o=element('option','',r.name,select);o.value=key;}select.value=value;return select;}
function renderSpecial(){
  const forced=$('forced-actions');forced.replaceChildren();forced.hidden=true;
  if(table.human_turn&&mode==='DISCARD_RESOURCE'){forced.hidden=false;for(const a of legal('DISCARD_RESOURCE'))button(forced,`${resourceName(a.value)} (${own().resources[a.value]})`,()=>perform(a.index),'resource-choice');}
  if(victimChoices.length){forced.hidden=false;for(const a of victimChoices)button(forced,a.value[1]?playerName(a.value[1]):'No theft',()=>selectBoardAction(a.index),'resource-choice');}
  const box=$('special-actions');box.replaceChildren();box.hidden=!selectedDev;if(!selectedDev)return;
  const available=legal(selectedDev);if(!available.length){selectedDev=null;box.hidden=true;return;}
  if(selectedDev==='PLAY_MONOPOLY'){element('span','parameter-label','Take all of one resource:',box);for(const a of available)button(box,resourceName(a.value),()=>perform(a.index),'resource-choice');}
  else if(selectedDev==='PLAY_YEAR_OF_PLENTY'){
    const row=element('div','dev-resource-row',undefined,box),first=resourceSelect(row,'dev-first',available[0].value[0]),second=resourceSelect(row,'dev-second',available[0].value[1]);
    const play=button(row,'Play',()=>{const a=find();if(a)perform(a.index);},'primary');
    const find=()=>available.find(a=>[...a.value].sort().join(',')===[first.value,second.value].sort().join(','));
    first.onchange=second.onchange=()=>{play.disabled=!find()||busy;};
  }
}
function eventText(event){
  const name=event.actor===table.human?'You':playerName(event.actor),past={BUILD_SETTLEMENT:'built a settlement',BUILD_ROAD:'built a road',BUILD_CITY:'upgraded a settlement to a city',END_TURN:'ended the turn',PLAY_KNIGHT_CARD:'played a Knight',PLAY_ROAD_BUILDING:'played Road Building',PLAY_MONOPOLY:'played Monopoly',PLAY_YEAR_OF_PLENTY:'played Year of Plenty',MOVE_ROBBER:'moved the robber',DISCARD_RESOURCE:'discarded a resource'};
  if(event.type==='ROLL')return`${name} rolled ${event.result.join(' + ')} = ${event.result.reduce((a,b)=>a+b,0)}.`;
  if(event.type==='BUY_DEVELOPMENT_CARD')return`${name} bought ${event.result?devNames[event.result]:'a development card'}.`;
  if(event.type==='MARITIME_TRADE'){const offered=event.value.slice(0,-1).filter(Boolean);return`${name} traded ${offered.length} ${resourceName(offered[0])} for 1 ${resourceName(event.value.at(-1))}.`;}
  return`${name} ${past[event.type]||event.type.toLowerCase().replaceAll('_',' ')}.`;
}
function renderLog(){
  $('log').replaceChildren();for(const event of [...table.log].reverse().slice(0,30)){const li=element('li','',eventText(event),$('log'));li.style.setProperty('--player',palette[event.actor]);}
  $('latest-event').textContent=table.log.length?eventText(table.log.at(-1)):'Place your first settlement.';
}
function renderDice(values){
  const rolled=Array.isArray(values)&&values.length===2&&values.every(n=>Number.isInteger(n)&&n>=1&&n<=6);
  const faces=rolled?values:[null,null],total=rolled?values[0]+values[1]:null;
  $('dice').replaceChildren();
  for(const n of faces){
    const face=svg('svg',{viewBox:'0 0 60 60',class:'die','data-value':n??'','aria-hidden':'true'},$('dice'));
    if(n===null){svgText(face,30,40,'–','die-empty');continue;}
    const points=[];
    if(n%2)points.push([30,30]);
    if(n>=2)points.push([15,15],[45,45]);
    if(n>=4)points.push([45,15],[15,45]);
    if(n===6)points.push([15,30],[45,30]);
    for(const[cx,cy]of points)svg('circle',{cx,cy,r:5.5,class:'die-pip'},face);
  }
  $('roll-total').textContent=total===null?'–':String(total);
  $('roll-note').textContent=!rolled?'No roll yet':total===7?'Robber roll':'';
  $('roll-note').hidden=rolled&&total!==7;
  $('roll-display').classList.toggle('is-seven',total===7);
  $('roll-display').setAttribute('aria-label',rolled?`Last roll: ${values[0]} plus ${values[1]} equals ${total}${total===7?'. Robber roll.':''}`:'No roll yet');
}
function acceptTable(data){if(table?.id!==data.id){previousHand=null;changes={};selectedDev=null;}table=data;retryNeeded=false;if(!legal(mode).length){mode=legal('MOVE_ROBBER').length?'MOVE_ROBBER':legal('DISCARD_RESOURCE').length?'DISCARD_RESOURCE':table.free_roads_available&&legal('BUILD_ROAD').length?'BUILD_ROAD':table.initial_phase?table.legal_actions[0]?.type:null;}victimChoices=[];if(pendingPlacement?.revision!==table.revision)pendingPlacement=null;render();}
function render(){
  document.body.classList.add('game-active');$('setup').hidden=true;$('dashboard').hidden=false;$('opponent-rail').hidden=false;$('new-game').hidden=false;
  $('match-label').textContent=`${table.players.length} players · ${own().actual_vp}/${table.target_vp} VP`;
  $('board-status').textContent=table.winner?`${playerName(table.winner)} wins`:table.initial_phase?'Initial placement':`${playerName(table.current)}'s turn`;
  $('turn-counter').textContent=`Turn ${table.turn+1}`;$('busy-chip').hidden=table.human_turn||!!table.winner||retryNeeded;
  $('board-tip').textContent=table.winner?'Game finished.':table.human_turn&&spatial.has(mode)?'Select a location. Confirm to build.':table.human_turn?'Your turn.':'Waiting for opponent.';
  $('deck-count').hidden=false;$('deck-count').textContent=`${table.development_deck_size} dev. cards left`;
  renderDice(table.dice);
  renderPlayers();renderHand();renderTurn();renderPurchases();renderTrade();renderDevelopment();renderSpecial();renderLog();drawBoard(table.geometry,table);scheduleBot();
}
function scheduleBot(){clearTimeout(botTimer);if(table&&!table.human_turn&&!table.winner&&!busy&&!retryNeeded)botTimer=setTimeout(()=>perform(null,true),350);}
async function perform(index,advance=false){
  if(busy||!table)return;const id=table.id;busy=true;pendingPlacement=null;selectedDev=null;victimChoices=[];clearTimeout(botTimer);renderTurn();renderPurchases();renderTrade();renderDevelopment();renderSpecial();drawTargets();
  try{const result=await api(advance?'/api/advance':'/api/action',{id,revision:table.revision,index});if(table?.id!==id)return;busy=false;acceptTable(result);}
  catch(e){if(table?.id!==id)return;busy=false;if(e.status===409){try{acceptTable(await api(`/api/game/${id}`));}catch{notify('Could not refresh the game. Reload the page.');}}else{notify(e.message);retryNeeded=true;render();clearTimeout(botTimer);}}
}
async function startGame(){if(busy)return;busy=true;$('start-game').disabled=true;$('start-game').textContent='Starting…';try{const data=await api('/api/new',{name:$('human-name').value.trim()||'You',bots});remember(data.id);busy=false;mode=null;acceptTable(data);}catch(e){busy=false;notify(e.message);}finally{$('start-game').disabled=false;$('start-game').textContent='Start game';}}
async function boot(){try{const data=await api('/api/bootstrap');renderBotCount();drawBoard(data.preview);for(const id of['trade-give','trade-get']){const select=$(id);select.replaceChildren();for(const[key,r]of Object.entries(resources)){const option=element('option','',r.name,select);option.value=key;}}let saved;try{saved=localStorage.getItem(storageKey);}catch{}if(saved){try{acceptTable(await api(`/api/game/${saved}`));}catch(e){if(e.status===404)remember(null);else notify(e.message);}}}catch{notify(globalThis.catanOfflineApi?'Could not load the game. Reload the page to retry.':'Could not connect to the game server. Reload the page.');$('start-game').disabled=true;}}
document.querySelectorAll('[data-count]').forEach(e=>e.onclick=()=>{bots=Array.from({length:+e.dataset.count},()=>'402');renderBotCount();});
$('start-game').onclick=startGame;$('rules-open').onclick=()=>$('rules').showModal();document.querySelectorAll('.rules-close').forEach(e=>e.onclick=()=>$('rules').close());
$('new-game').onclick=()=>$('new-dialog').showModal();$('cancel-new').onclick=()=>$('new-dialog').close();
$('confirm-new').onclick=async()=>{clearTimeout(botTimer);table=null;busy=false;pendingPlacement=null;mode=null;selectedDev=null;victimChoices=[];previousHand=null;changes={};remember(null);$('new-dialog').close();document.body.classList.remove('game-active');$('setup').hidden=false;$('dashboard').hidden=true;$('opponent-rail').hidden=true;$('new-game').hidden=true;$('busy-chip').hidden=true;$('match-label').textContent='Play against bots';$('board-status').textContent='Board preview';$('turn-counter').textContent='First to 10 points';$('board-tip').textContent='Choose your opponents to start.';$('deck-count').hidden=true;renderDice(null);await boot();};
$('confirm-placement').onclick=()=>pendingPlacement?perform(pendingPlacement.index):undefined;
$('cancel-placement').onclick=()=>{pendingPlacement=null;renderTurn();drawTargets();};
$('retry-bot').onclick=()=>{retryNeeded=false;perform(null,true);};
$('trade-give').onchange=()=>{tradeGive=$('trade-give').value;if(tradeGet===tradeGive)tradeGet=Object.keys(resources).find(r=>r!==tradeGive);renderTrade();};
$('trade-get').onchange=()=>{tradeGet=$('trade-get').value;renderTrade();};
$('trade-submit').onclick=()=>{const a=tradeAction();if(a&&!busy)return perform(a.index);};
$('board').addEventListener('click',selectNearest);
renderDice(null);
boot();
