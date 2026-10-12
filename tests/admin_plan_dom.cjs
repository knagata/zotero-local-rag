const assert = require('node:assert/strict');
const vm = require('node:vm');
async function testAdminPlan(source) {
  class Element {
    constructor(tag='div'){this.tag=tag;this.children=[];this.handlers={};this.checked=false;this.disabled=false;this.textContent='';this.style={};}
    append(...children){this.children.push(...children);}
    replaceChildren(...children){this.children=children;}
    addEventListener(type,handler){this.handlers[type]=handler;}
    setAttribute(){}
    showModal(){this.open=true;}
    close(value){this.open=false;this.returnValue=value;return this.handlers.close?.();}
  }
  const elements=new Map(), posts=[];
  const stages=[
    {key:'library_update',label:'Library',requires:[],default:true,paid:true},
    {key:'structure_update',label:'Structure',requires:['library_update'],default:true,paid:true},
    {key:'database_audit',label:'Audit',requires:[],default:true,paid:false},
    {key:'summary_batch',label:'Summary',requires:['structure_update','database_audit'],default:false,paid:true},
    {key:'citation_update',label:'Citations',requires:[],default:true,paid:true},
  ];
  const data={stages,manifest:{},artifacts:{},database_gate:{},jobs:[]};
  const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(tag){return new Element(tag);}};
  const context=vm.createContext({document,setInterval(){},confirm(){return true;},location:{},console,fetch:async(path,options)=>{
    if(options.method==='POST')posts.push(JSON.parse(options.body));
    return {ok:true,status:200,headers:{get(){return 'application/json';}},json:async()=>data};
  }});
  vm.runInContext(source,context);await vm.runInContext('refresh()',context);
  const actions=()=>document.getElementById('actions').children;
  const box=key=>actions()[stages.findIndex(s=>s.key===key)].children[0];
  assert.equal(box('library_update').checked,true);
  assert.equal(box('summary_batch').checked,false);
  box('library_update').checked=false;box('library_update').handlers.change();
  assert.equal(box('structure_update').disabled,true);
  assert.equal(box('structure_update').checked,false);
  assert.equal(box('summary_batch').disabled,true);
  await vm.runInContext('refresh()',context);assert.equal(box('library_update').checked,false);
  actions().at(-1).handlers.click(); // restore quick selection
  actions().at(-2).handlers.click(); // open paid confirmation
  assert.equal(document.getElementById('confirm-consent').checked,false);
  assert.equal(document.getElementById('confirm-button').disabled,true);
  document.getElementById('confirm-form').handlers.submit({preventDefault(){}});
  assert.equal(posts.length,0);
  document.getElementById('confirm-consent').checked=true;
  document.getElementById('confirm-consent').handlers.change();
  assert.equal(document.getElementById('confirm-button').disabled,false);
  await document.getElementById('confirm-dialog').close('confirm');
  assert.equal(posts.length,1);assert.equal(posts[0].paid_consent,true);
  assert.deepEqual(posts[0].selected_steps,['library_update','structure_update','database_audit','citation_update']);
  actions().at(-2).handlers.click();assert.equal(document.getElementById('confirm-consent').checked,false);
  assert.equal(document.getElementById('confirm-dialog').returnValue,'');
  await document.getElementById('confirm-dialog').close('cancel');assert.equal(posts.length,1);
  delete data.stages;await vm.runInContext('refresh()',context);
  assert.match(actions()[0].textContent,/工程一覧を取得できません/);
  assert.match(actions()[1].textContent,/manage_citation_graph_service.py restart/);
  assert.equal(actions().some(x=>x.tag==='button'),false);
  data.stages=stages;await vm.runInContext('refresh()',context);
  assert.equal(box('library_update').checked,true);
  return 'checkbox dependency, polling, default, consent, cancel and old-server warning passed';
}
module.exports = testAdminPlan;
if(require.main===module)testAdminPlan(require('node:fs').readFileSync('citation_graph/static/admin.js','utf8')).then(console.log);
