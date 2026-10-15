const assert=require('node:assert/strict');
const vm=require('node:vm');
async function testPdfReview(source){
 class Element{
  constructor(){this.style={setProperty(name,value){this[name]=value;}};this.children=[];this.width=1;this.height=1;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  getContext(){return {};}
  remove(){this.removed=true;}
 }
 const document={createElement(){return new Element();}},context=vm.createContext({document,window:{},devicePixelRatio:2,console});
 vm.runInContext(source,context);
 let loads=0,destroyed=0,textLayers=0,cancelled=0;const options=[],rendered=[];
 const library={GlobalWorkerOptions:{},AnnotationMode:{DISABLE:0},getDocument(option){loads++;options.push(option);return {promise:Promise.resolve({numPages:10,getPage:async(number)=>({userUnit:1,getViewport({scale}){return {width:600*scale,height:900*scale,scale};},render(args){rendered.push(args);return {promise:Promise.resolve(),cancel(){cancelled++;}};},streamTextContent(){return {items:[{str:'Selectable original text'}]};}})}),destroy:async()=>{destroyed++;}};},TextLayer:class{constructor({container,textContentSource}){textLayers++;container.textContent=textContentSource.items[0].str;}render(){return Promise.resolve();}cancel(){}}};
 const stage=new Element(),viewport={clientWidth:500,clientHeight:700};
 const viewer=new context.window.ReviewPdf(stage,viewport,async()=>library);
 await viewer.open('/doc');await viewer.open('/doc');assert.equal(loads,1);
 assert.ok(options[0].disableAutoFetch&&options[0].disableStream&&!options[0].isEvalSupported);
 assert.equal(library.GlobalWorkerOptions.workerSrc,'/admin/pdfjs/pdf.worker.mjs');
 for(const [width,height] of [[800,900],[500,500],[240,360]]){
  viewport.clientWidth=width;viewport.clientHeight=height;
  await viewer.show(1,'fit',100);const sheet=stage.children[0];
  assert.ok(parseFloat(sheet.style.width)<=width&&parseFloat(sheet.style.height)<=height);
  assert.equal(sheet.children[1].textContent,'Selectable original text');
 }
 await viewer.show(1,'custom',400);
 assert.ok(parseFloat(stage.style.width)>viewport.clientWidth);
 const canvas=stage.children[0].children[0];assert.ok(canvas.width*canvas.height<=4000000);
 viewer.prefetch(1,'fit',100);await new Promise(resolve=>setImmediate(resolve));
 for(let page=2;page<=8;page++)await viewer.show(page,'fit',100);
 assert.ok(viewer.cache.size<=3);assert.ok(cancelled>0&&textLayers>0&&rendered.length>0);
 await viewer.open('/other');assert.equal(loads,2);assert.equal(destroyed,1);
 viewer.clear();assert.equal(viewer.cache.size,0);assert.equal(stage.children.length,0);
 // A slow previous draw must never replace the new page.
 await viewer.open('/third');let finish;
 viewer.render=async(number)=>number===1?new Promise(resolve=>finish=resolve):{sheet:Object.assign(new Element(),{page:number}),width:100,height:100,scale:1};
 const slow=viewer.show(1,'fit',100);await viewer.show(2,'fit',100);finish({sheet:new Element(),width:100,height:100,scale:1});await slow;
 assert.equal(stage.children[0].page,2);
 console.log('PDF renderer: reusable handle, selectable text layer, fit/zoom, bounded prefetch and stale draws passed');
}
module.exports={testPdfReview};
if(require.main===module)testPdfReview(require('node:fs').readFileSync('citation_graph/static/debug-pdf.js','utf8')).catch(error=>{console.error(error);process.exitCode=1;});
