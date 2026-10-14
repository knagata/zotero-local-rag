const assert=require('node:assert/strict');
const vm=require('node:vm');
async function testDebugReview(source){
 class Element{
  constructor(tag='div'){this.tag=tag;this.children=[];this.handlers={};this.value='';this.checked=false;this.disabled=false;this.hidden=false;this.textContent='';this.style={};this.tagName=tag.toUpperCase();}
  focus(){}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=items;}
  addEventListener(event,callback){this.handlers[event]=callback;}
  setAttribute(){}
 }
 const elements=new Map(),posts=[];
 const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(tag){return new Element(tag);}};
 const html=require('node:fs').readFileSync(require.resolve('../citation_graph/static/debug.html'),'utf8');
 for(const match of html.matchAll(/id="([^"]+)"/g))document.getElementById(match[1]);
 for(const match of source.matchAll(/\$\('([^']+)'\)/g))assert.ok(html.includes(`id="${match[1]}"`),`missing control ${match[1]}`);
 const windowHandlers={};const version='a'.repeat(64);
 const page={attachment_key:'ABCD1234',version,page:1,total_pages:2,title:'Book',anchors:[{id:'C1',page:1,zone:'body'},{id:'C2',page:2,zone:'endnote'}],current_toc:[],draft:{revision:0,edits:{},toc:[]},chunks:[{id:'C1',text:'Original',metadata:{page:1}}]};
 let revision=0,edits={},originals={},pendingSave=null,deferSave=false;
 const context=vm.createContext({document,window:{addEventListener(type,handler){windowHandlers[type]=handler;}},confirm:()=>true,console,fetch:async(path,options={})=>{
  let data;
  if(path.endsWith('/documents'))data=[{attachment_key:'ABCD1234',title:'Book',issues:['OCR']}];
  else if(path.includes('/pages/'))data={...page,page:Number(path.split('/').at(-1))};
  else if(path.endsWith('/draft')){
   const body=JSON.parse(options.body);posts.push(body);edits=body.edits;originals={C1:'Original'};
   data={revision:++revision,...body};data.revision=revision;
   if(deferSave)await new Promise(resolve=>pendingSave=resolve);
  }else if(path.includes('/diff?'))data={revision,text_changes:[{chunk_id:'C1',before:'Original',after:edits.C1}],toc_before:[],toc_after:[]};
  else if(path.includes('/restore?')){posts.push(JSON.parse(options.body));data={revision:++revision,edits:originals,toc:[],toc_changed:false};}
  else if(path.endsWith('/history'))data=revision?[{revision,created_at:'2026-10-15',actor:'reviewer',text_changes:1,headings:0}]:[];
  else if(path.endsWith('/apply')){posts.push(JSON.parse(options.body));data={id:'job'};}
  else data={revision,edits,originals};
  return {ok:true,status:200,json:async()=>data};
 }});
 vm.runInContext(source,context);await new Promise(resolve=>setImmediate(resolve));
 await vm.runInContext("selectDocument('ABCD1234')",context);
 assert.equal(elements.get('document-panel').hidden,true);
 const chunk=elements.get('chunks').children[0];const input=chunk.children.find(e=>e.tag==='textarea');
 input.value='Corrected';input.handlers.input();
 assert.equal(vm.runInContext('state.dirty',context),true);
 await vm.runInContext('save()',context);
 assert.equal(posts[0].edits.C1,'Corrected');assert.equal(vm.runInContext('state.revision',context),1);
 assert.equal(vm.runInContext('state.dirty',context),false);
 await elements.get('apply').handlers.click(); // first click shows saved diff, no DB request
 assert.equal(posts.length,1);
 await elements.get('apply').handlers.click();
 assert.equal(posts.length,2);assert.equal(posts[1].revision,1);
 await vm.runInContext("restoreHistory('ABCD1234',1,true)",context);
 assert.equal(vm.runInContext('state.revision',context),2);
 assert.equal(vm.runInContext('state.edits.C1',context),'Original');
 // Work areas switch without page scrolling; unsaved text survives switching.
 vm.runInContext("setTab('notes')",context);
 assert.equal(elements.get('panel-body').hidden,true);assert.equal(elements.get('panel-notes').hidden,false);
 let noteInput=elements.get('notes-chunks').children[0].children.find(e=>e.tag==='textarea');
 noteInput.value='Note correction';noteInput.handlers.input();
 vm.runInContext("setTab('body')",context);
 assert.equal(elements.get('chunks').children[0].children.find(e=>e.tag==='textarea').value,'Note correction');
 elements.get('zone-start').value=1;elements.get('zone-end').value=2;elements.get('zone-kind').value='footnote';
 elements.get('assign-zone').handlers.click();
 assert.equal(vm.runInContext('state.zones.C2',context),'footnote');
 // Headings can be added independently of a chunk button.
 elements.get('add-heading').handlers.click();
 assert.equal(vm.runInContext('state.tab',context),'toc');
 const headingRow=elements.get('toc').children[0];
 const headingTitle=headingRow.children[1].children[0];headingTitle.value='A chapter';headingTitle.handlers.input();
 assert.equal(vm.runInContext('state.toc[0].title',context),'A chapter');
 // Fit is bounded in both dimensions across viewport sizes. Zoom enables overflow.
 const image=elements.get('paper'),viewport=elements.get('paper-viewport');image.naturalWidth=1200;image.naturalHeight=1800;
 for(const [width,height] of [[800,900],[500,500],[240,360]]){viewport.clientWidth=width;viewport.clientHeight=height;elements.get('zoom-mode').value='fit';vm.runInContext('fitPaper()',context);assert.ok(parseFloat(image.style.width)<=width);assert.ok(parseFloat(image.style.height)<=height);}
 elements.get('zoom').value=100;elements.get('zoom-mode').value='custom';vm.runInContext('fitPaper()',context);
 assert.equal(image.style.width,'1200px');assert.equal(elements.get('paper-stage').style.width,'1200px');
 assert.equal(elements.get('previous').disabled,true);assert.equal(elements.get('next').disabled,false);
 await elements.get('next').handlers.click();assert.equal(vm.runInContext('state.page',context),2);
 assert.equal(elements.get('next').disabled,true);assert.equal(elements.get('previous').disabled,false);
 // Cursor keys must not change pages while editing text.
 windowHandlers.keydown({key:'ArrowLeft',target:{tagName:'TEXTAREA'}});
 assert.equal(vm.runInContext('state.page',context),2);
 // Edits made while a save is in flight must remain visibly unsaved.
 deferSave=true;var saving=vm.runInContext('save()',context);
 await new Promise(resolve=>setImmediate(resolve));
 vm.runInContext("state.edits.C1='Later';markDirty()",context);
 pendingSave();await saving;
 assert.equal(vm.runInContext('state.dirty',context),true);
 assert.ok(elements.get('notice').textContent.includes('未保存'));
 // An old save response must not overwrite another document's revision.
 var oldSave=vm.runInContext('save()',context);await new Promise(resolve=>setImmediate(resolve));
 vm.runInContext("state.key='OTHER123';state.revision=0;state.request++",context);
 pendingSave();await oldSave;
 assert.equal(vm.runInContext('state.revision',context),0);
 console.log('debug review DOM: history, diff, apply and asynchronous save guards passed');
}
module.exports={testDebugReview};
if(require.main===module)testDebugReview(require('node:fs').readFileSync('citation_graph/static/debug.js','utf8')).catch(error=>{console.error(error);process.exitCode=1;});
