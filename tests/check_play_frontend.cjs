/* Application-code integration against the real HTTP server, plus isolated
 * view fixtures for bank trades and development-card parameter selection.
 * No browser is launched: this does not validate rendered pixels/layout. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
class Element{
  constructor(tag='div'){this.tagName=tag;this.children=[];this.attributes={};this._text='';this.value='';this.hidden=false;this.disabled=false;this.open=false;this.className='';this.listeners={};this.style={setProperty(){}};this.classList={toggle:(k,on)=>{const s=new Set(this.className.split(' ').filter(Boolean));on?s.add(k):s.delete(k);this.className=[...s].join(' ');},add:k=>this.classList.toggle(k,true),remove:k=>this.classList.toggle(k,false),contains:k=>this.className.split(' ').includes(k)};}
  append(e){this.children.push(e);e.parent=this;}appendChild(e){this.append(e);return e;}
  replaceChildren(...els){this.children=[];this._text='';els.forEach(e=>this.append(e));}
  setAttribute(k,v){this.attributes[k]=String(v);}getAttribute(k){return this.attributes[k];}
  addEventListener(type,fn){(this.listeners[type]??=[]).push(fn);}
  getBoundingClientRect(){return{left:0,top:0,width:390,height:350};}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}set textContent(v){this._text=String(v);this.children=[];}
  showModal(){this.open=true;}close(){this.open=false;}
}
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(path.join(root,'src/catanzero/play_web/index.html'),'utf8');
const matches=[...html.matchAll(/id="([^"]+)"/g)],elements=new Map(matches.map(m=>[m[1],new Element()]));
assert.equal(elements.size,matches.length,'unique HTML IDs');
assert(html.includes('lang="en" dir="ltr"'));
assert(!/[\u0590-\u05ff]/u.test(html),'all fixed interface copy is English');
assert(!html.includes('zoom-in')&&!html.includes('sheet-backdrop'),'no zoom controls or mobile drawers');
const counts=[1,2,3].map(n=>Object.assign(new Element('button'),{dataset:{count:String(n)}})),closes=[new Element('button'),new Element('button')];
const body=new Element('body'),document={body,getElementById:id=>{assert(elements.has(id),`unknown element ${id}`);return elements.get(id);},createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t),querySelectorAll:s=>s==='[data-count]'?counts:s==='.rules-close'?closes:[]};
const storage=new Map(),timers=new Map();let timerId=0;
const base=(process.argv[2]||'http://127.0.0.1:8787').replace(/\/$/,'');
const context=vm.createContext({document,console,fetch:(p,o)=>fetch(base+p,o),localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},setTimeout:fn=>{timers.set(++timerId,fn);return timerId;},clearTimeout:id=>timers.delete(id)});
const read=code=>vm.runInContext(code,context);
async function until(fn){for(let i=0;i<500;i++){if(fn())return;await new Promise(r=>setTimeout(r,10));}throw Error('initialization timed out');}
function verifyFit(){
  const box=read('fitBox'),geometry=read('table.geometry');
  for(const p of Object.values(geometry.nodes))assert(p[0]>=box.x&&p[0]<=box.x+box.width&&p[1]>=box.y&&p[1]<=box.y+box.height,'all intersections inside fitted board');
  for(const t of geometry.tiles.filter(t=>t.kind==='port')){context.fitPort=t;const p=read('portPosition(fitPort,table.geometry)');assert(p.x-17>=box.x&&p.x+17<=box.x+box.width&&p.y-17>=box.y&&p.y+17<=box.y+box.height,'complete port inside fitted board');}
}
async function main(){
  vm.runInContext(fs.readFileSync(path.join(root,'src/catanzero/play_web/app.js'),'utf8'),context);
  await until(()=>elements.get('trade-get').children.length===5);
  assert(!html.includes('bot-selectors'),'no difficulty selector');
  assert(html.includes('All bots use Model 402.'));
  assert.equal(elements.get('tiles').children.length,19);assert.equal(elements.get('ports').children.filter(e=>e.tagName==='g').length,9);
  assert.equal(elements.get('roll-total').textContent,'–');
  for(let n=1;n<=6;n++){
    read(`renderDice([${n},${7-n}])`);
    const faces=elements.get('dice').children;
    assert.equal(faces.length,2);
    assert.equal(faces[0].children.filter(e=>e.tagName==='circle').length,n,'first die pip count');
    assert.equal(faces[1].children.filter(e=>e.tagName==='circle').length,7-n,'second die pip count');
    assert.equal(elements.get('roll-total').textContent,'7');
    assert(elements.get('roll-display').classList.contains('is-seven'));
    assert.equal(elements.get('roll-note').textContent,'Robber roll');
  }
  read('renderDice([6,6])');assert.equal(elements.get('roll-total').textContent,'12');
  assert.equal(elements.get('roll-note').hidden,true);assert(!elements.get('roll-display').classList.contains('is-seven'));
  read('renderDice(null)');assert.equal(elements.get('roll-total').textContent,'–');
  counts[2].onclick();assert.equal(read('bots.join(",")'),'402,402,402');
  counts[1].onclick();assert.equal(read('bots.join(",")'),'402,402');
  await elements.get('start-game').onclick();assert.equal(read('table.players.length'),3);
  assert(read('table.players.filter(p=>!p.human).every(p=>p.model==="402")'));
  elements.get('new-game').onclick();await elements.get('confirm-new').onclick();
  counts[0].onclick();elements.get('human-name').value='Frontend test';await elements.get('start-game').onclick();
  assert.equal(read('table.players.length'),2);assert.equal(read('table.players[1].model'),'402');
  assert.equal(elements.get('dashboard').hidden,false);assert(body.classList.contains('game-active'));
  assert.equal(elements.get('hand').children.length,5);assert.equal(elements.get('purchases').children.length,4);
  assert.equal(elements.get('purchases').children[0].disabled,true);assert.equal(elements.get('purchases').children[1].disabled,false);
  assert.equal(elements.get('trade-submit').disabled,true);verifyFit();
  // Map taps only preview. Cancel must not send a move; confirm sends one.
  elements.get('targets').children[0].onclick();assert.equal(read('table.revision'),0);assert.equal(elements.get('buildings').children.length,0);
  assert.equal(elements.get('placement-panel').hidden,false);elements.get('cancel-placement').onclick();assert.equal(read('pendingPlacement'),null);
  elements.get('targets').children[0].onclick();await elements.get('confirm-placement').onclick();assert.equal(read('table.revision'),1);assert.equal(elements.get('buildings').children.length,1);
  assert(elements.get('targets').children[0].children.some(e=>e.getAttribute('class')==='target-road-border'));
  elements.get('targets').children[0].onclick();await elements.get('confirm-placement').onclick();assert.equal(read('table.revision'),2);
  assert(elements.get('roads').children.some(e=>e.getAttribute('class')==='road-piece'&&e.getAttribute('stroke-width')==='10'));
  let liveDecisions=0;
  for(let i=0;i<120&&!read('table.winner');i++){
    if(read('table.human_turn'))await read('perform(table.legal_actions[0].index)');else await read('perform(null,true)');
    liveDecisions++;
    assert.equal(elements.get('toast').textContent,'');
    assert.equal(elements.get('purchases').children.length,4,'all purchase prices stay present');
    if(read('!!table.dice')){
      assert.equal(elements.get('roll-total').textContent,String(read('table.dice[0]+table.dice[1]')),'total agrees with real engine roll');
      assert.equal(elements.get('dice').children.map(e=>e.getAttribute('data-value')).join(','),read('table.dice.join(",")'));
    }
  }
  assert(read('table.revision>=122||!!table.winner'));assert.equal(read('table.initial_phase'),false);assert(elements.get('log').children.length>0);
  assert(!/[\u0590-\u05ff]/u.test(elements.get('players').textContent));
  // UI-only fixtures, not engine gameplay: verify resource gains and legal
  // trade mapping without sending artificial actions to the real server.
  read('globalThis.savedTable=table; globalThis.originalPerform=perform; globalThis.sentIndex=null; perform=async index=>{sentIndex=index;}; table=JSON.parse(JSON.stringify(table)); table.human_turn=true; table.initial_phase=false; table.prompt="PLAY_TURN"; mode=null; table.free_roads_available=0; own().HAS_ROLLED=false; own().resources.WOOD=4; table.legal_actions=[{index:1,type:"ROLL",value:null}];tradeGive="WOOD";tradeGet="ORE";renderTrade();');
  assert.equal(elements.get('trade-submit').disabled,true);
  assert.equal(elements.get('trade-submit').textContent,'Roll first');
  assert.equal(elements.get('trade-message').textContent,'Roll the dice before trading.');
  read('own().HAS_ROLLED=true;table.legal_actions=[{index:887,type:"MARITIME_TRADE",value:["WOOD","WOOD","WOOD","WOOD","ORE"]}];');
  read('previousHand={...own().resources,WOOD:3};renderHand();renderTrade();');
  assert(elements.get('hand').children[0].textContent.includes('+1'));
  assert.equal(elements.get('trade-submit').disabled,false);await elements.get('trade-submit').onclick();assert.equal(read('sentIndex'),887);
  elements.get('trade-get').value='WOOD';elements.get('trade-get').onchange();
  assert.equal(read('tradeGive'),'WOOD');assert.equal(elements.get('trade-submit').disabled,true);
  assert.equal(elements.get('trade-message').textContent,'Choose a different resource to receive.');
  read('tradeGet="ORE";table.legal_actions=[];table.resource_bank.ORE=0;renderTrade();');
  assert.equal(elements.get('trade-message').textContent,'The bank has no Ore. Choose another resource.');
  read('table.resource_bank.ORE=19;own().resources.WOOD=3;renderTrade();');
  assert(elements.get('trade-message').textContent.includes('4 of the same resource'));
  read('own().resources.WOOD=4;table.legal_actions=[{index:887,type:"MARITIME_TRADE",value:["WOOD","WOOD","WOOD","WOOD","ORE"]}];');
  read('tradeGet="BRICK";renderTrade();');assert.equal(elements.get('trade-submit').disabled,true);
  read('tradeGet="ORE";table.buildings=[[table.geometry.tiles.find(t=>t.kind==="port"&&t.resource==="WOOD").edge[0],table.human,"SETTLEMENT"]];table.legal_actions=[{index:888,type:"MARITIME_TRADE",value:["WOOD","WOOD",null,null,"ORE"]}];renderTrade();');
  assert.equal(elements.get('trade-submit').textContent,'Trade 2:1');assert.equal(elements.get('trade-submit').disabled,false);await elements.get('trade-submit').onclick();assert.equal(read('sentIndex'),888);
  read('table.buildings=[[table.geometry.tiles.find(t=>t.kind==="port"&&!t.resource).edge[0],table.human,"SETTLEMENT"]];table.legal_actions=[{index:889,type:"MARITIME_TRADE",value:["WOOD","WOOD","WOOD",null,"ORE"]}];renderTrade();');
  assert.equal(elements.get('trade-submit').textContent,'Trade 3:1');assert.equal(elements.get('trade-submit').disabled,false);
  read('table.legal_actions=[{index:80,type:"PLAY_MONOPOLY",value:"ORE"},{index:81,type:"PLAY_MONOPOLY",value:"WOOD"}]; selectedDev="PLAY_MONOPOLY"; renderSpecial();');
  await elements.get('special-actions').children[1].onclick();assert.equal(read('sentIndex'),80);
  read('table.legal_actions=[{index:90,type:"PLAY_YEAR_OF_PLENTY",value:["WOOD","ORE"]},{index:91,type:"PLAY_YEAR_OF_PLENTY",value:["SHEEP","SHEEP"]}];selectedDev="PLAY_YEAR_OF_PLENTY";renderSpecial();');
  const row=elements.get('special-actions').children[0],[first,second,play]=row.children;
  first.value='BRICK';first.onchange();assert.equal(play.disabled,true);
  first.value='SHEEP';second.value='SHEEP';second.onchange();assert.equal(play.disabled,false);await play.onclick();assert.equal(read('sentIndex'),91);
  read('table.legal_actions=[{index:95,type:"DISCARD_RESOURCE",value:"WOOD"}];mode="DISCARD_RESOURCE";renderSpecial();');
  assert.equal(elements.get('forced-actions').hidden,false);await elements.get('forced-actions').children[0].onclick();assert.equal(read('sentIndex'),95);
  read('perform=originalPerform;table=savedTable;selectedDev=null;mode=null;render();');
  const id=read('table.id'),revision=read('table.revision');await read('boot()');assert.equal(read('table.id'),id);assert.equal(read('table.revision'),revision);
  elements.get('new-game').onclick();await elements.get('confirm-new').onclick();assert.equal(read('table'),null);assert.equal(body.classList.contains('game-active'),false);
  assert.equal(elements.get('roll-total').textContent,'–');assert.equal(elements.get('roll-note').textContent,'No roll yet');
  counts[2].onclick();await elements.get('start-game').onclick();assert.equal(elements.get('players').children.length,4);verifyFit();
  assert(read('table.players.filter(p=>!p.human).every(p=>p.model==="402")'));
  assert(!/Easy|Medium|Hard/.test(elements.get('players').textContent),'no difficulty labels on player cards');
  elements.get('new-game').onclick();await elements.get('confirm-new').onclick();
  console.log(JSON.stringify({status:'passed',kind:'Node DOM + real HTTP, plus isolated view fixtures',live_decisions:liveDecisions,checks:'English copy; full board and ports fit; visible prices; preview/cancel/confirm; resource changes; dice faces 1–6, sum, 7, reset and engine agreement; 4:1/3:1/2:1 trades; development-card choices; discards; 2/4 players; resume',visual_browser_qa:false}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
