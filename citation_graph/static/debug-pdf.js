'use strict';
// One PDF handle and at most three rendered pages; all resources are same-origin.
class ReviewPdf {
 constructor(stage,viewport,loadLibrary=()=>import('/admin/pdfjs/pdf.mjs')){
  this.stage=stage;this.viewport=viewport;this.loadLibrary=loadLibrary;
  this.url=null;this.loading=null;this.pdf=null;this.library=null;this.serial=0;this.request=0;this.cache=new Map();
 }
 clear(){
  this.serial++;this.request++;this.url=null;this.pdf=null;this.stage.replaceChildren();
  for(const entry of this.cache.values())this.release(entry);this.cache.clear();
  const loading=this.loading;this.loading=null;if(loading)loading.destroy().catch(()=>{});
 }
 release(entry){entry.task?.cancel();entry.text?.cancel();if(entry.canvas){entry.canvas.width=0;entry.canvas.height=0;}entry.sheet?.remove();}
 async open(url){
  if(this.url===url&&this.ready)return this.ready;
  this.clear();this.url=url;const serial=this.serial;
  this.ready=(async()=>{
   const library=await this.loadLibrary();if(serial!==this.serial)return null;
   this.library=library;library.GlobalWorkerOptions.workerSrc='/admin/pdfjs/pdf.worker.mjs';
   this.loading=library.getDocument({url,withCredentials:true,isEvalSupported:false,
    disableStream:true,disableAutoFetch:true,rangeChunkSize:262144,canvasMaxAreaInBytes:16777216,
    cMapUrl:'/admin/pdfjs/cmaps/',cMapPacked:true,standardFontDataUrl:'/admin/pdfjs/standard_fonts/',
    wasmUrl:'/admin/pdfjs/wasm/',iccUrl:'/admin/pdfjs/iccs/'});
   const pdf=await this.loading.promise;if(serial!==this.serial)return null;
   this.pdf=pdf;return pdf;
  })();
  return this.ready;
 }
 scale(page,mode,zoom){
  const base=page.getViewport({scale:1}),width=Math.max(1,this.viewport.clientWidth),height=Math.max(1,this.viewport.clientHeight);
  const factor=mode==='fit'?Math.min(width/base.width,height/base.height):mode==='width'?width/base.width:Math.max(.05,Math.min(4,Number(zoom)/100||1))*4/3;
  return Math.floor(factor*10000)/10000;
 }
 async render(number,mode,zoom){
  const pdf=this.pdf;if(!pdf)return null;const serial=this.serial;
  const page=await pdf.getPage(number);if(serial!==this.serial)return null;
  const scale=this.scale(page,mode,zoom),ratio=Math.min(globalThis.devicePixelRatio||1,2);
  const key=`${number}:${scale}:${ratio}`;
  if(this.cache.has(key)){const cached=this.cache.get(key);this.cache.delete(key);this.cache.set(key,cached);return cached.promise;}
  const sheet=document.createElement('div');sheet.className='pdf-sheet';
  const canvas=document.createElement('canvas'),textContainer=document.createElement('div');textContainer.className='textLayer';
  const viewport=page.getViewport({scale});sheet.style.width=`${viewport.width}px`;sheet.style.height=`${viewport.height}px`;
  sheet.style.setProperty('--scale-factor',String(scale));sheet.style.setProperty('--user-unit',String(page.userUnit||1));sheet.style.setProperty('--total-scale-factor',String(scale*(page.userUnit||1)));
  const outputScale=Math.min(ratio,Math.sqrt(4000000/(viewport.width*viewport.height)));
  canvas.width=Math.max(1,Math.floor(viewport.width*outputScale));canvas.height=Math.max(1,Math.floor(viewport.height*outputScale));
  canvas.style.width=`${viewport.width}px`;canvas.style.height=`${viewport.height}px`;sheet.append(canvas,textContainer);
  const entry={sheet,canvas,scale,width:viewport.width,height:viewport.height,task:null,text:null,promise:null};
  entry.promise=(async()=>{
   entry.task=page.render({canvasContext:canvas.getContext('2d'),viewport,transform:[outputScale,0,0,outputScale,0,0]});
   await entry.task.promise;
   if(serial!==this.serial||this.cache.get(key)!==entry)throw new Error('古いPDF描画を取り消しました。');
   entry.text=new this.library.TextLayer({textContentSource:page.streamTextContent(),container:textContainer,viewport});
   await entry.text.render();return entry;
 })().catch(error=>{if(this.cache.get(key)===entry)this.cache.delete(key);this.release(entry);throw error;});
  this.cache.set(key,entry);
  while(this.cache.size>3){const oldest=this.cache.keys().next().value;const removed=this.cache.get(oldest);this.cache.delete(oldest);this.release(removed);}
  return entry.promise;
 }
 async show(number,mode,zoom){
  const request=++this.request;let entry;try{entry=await this.render(number,mode,zoom);}catch(error){if(request!==this.request)return null;throw error;}
  if(!entry||request!==this.request)return null;
  this.stage.style.width=`${Math.max(this.viewport.clientWidth,entry.width)}px`;this.stage.style.height=`${Math.max(this.viewport.clientHeight,entry.height)}px`;
  this.stage.replaceChildren(entry.sheet);if(mode==='fit'){this.viewport.scrollTop=0;this.viewport.scrollLeft=0;}
  return Math.round(entry.scale/(4/3)*100);
 }
 prefetch(number,mode,zoom){
  if(!this.pdf)return;
  for(const nearby of [number+1,number-1])if(nearby>=1&&nearby<=this.pdf.numPages)this.render(nearby,mode,zoom).catch(()=>{});
 }
}
window.ReviewPdf=ReviewPdf;
