'use strict';
const $=id=>document.getElementById(id);
const pdfViewer=new window.ReviewPdf($('paper-stage'),$('paper-viewport'));
const pageCache=new Map();
function cachedPage(key,page){const cacheKey=`${key}:${page}`;if(!pageCache.has(cacheKey)){const promise=api(`/admin/debug/api/documents/${key}/pages/${page}`).catch(error=>{if(pageCache.get(cacheKey)===promise)pageCache.delete(cacheKey);throw error;});pageCache.set(cacheKey,promise);while(pageCache.size>3)pageCache.delete(pageCache.keys().next().value);}return pageCache.get(cacheKey);}
function prefetchPages(){const keep=new Set([state.page,state.page-1,state.page+1].map(page=>`${state.key}:${page}`));for(const key of pageCache.keys())if(!keep.has(key))pageCache.delete(key);for(const page of [state.page+1,state.page-1])if(page>=1&&page<=state.totalPages)cachedPage(state.key,page).catch(()=>{});pdfViewer.prefetch(state.page,$('zoom-mode').value,$('zoom').value);}
const state={documents:[],key:null,page:1,version:null,revision:0,edits:{},toc:[],tocChanged:false,zones:{},pageChunks:[],allAnchors:[],tab:'body',totalPages:0,originals:new Map(),anchors:new Map(),dirty:false,generation:0,request:0,originalToc:[],reviewedRevision:null};
function node(tag,text=''){const e=document.createElement(tag);e.textContent=text;return e;}
function notice(text){$('notice').textContent=text;}
function markDirty(){state.dirty=true;state.generation++;state.reviewedRevision=null;}
async function api(path,options={}){const response=await fetch(path,{credentials:'same-origin',headers:{'Content-Type':'application/json'},...options});if(!response.ok){let text=`HTTP ${response.status}`;try{text=(await response.json()).error||text;}catch{}throw new Error(text);}return response.json();}
function draftStatus(){ $('draft-state').textContent=`下書き 第${state.revision}版 · 本文修正 ${Object.keys(state.edits).length}件 · 見出し構造${state.tocChanged?'変更あり':'変更なし'} · 分類 ${Object.keys(state.zones).length}件${state.dirty?' · 未保存':''}`;}
function issueLabel(reason){return ({flat_fallback:'目次の階層を復元できません',failed:'抽出・構造処理の失敗',blocked:'処理を保留中',degraded:'抽出・構造の要確認',awaiting_mistral_ocr_batch:'OCR待ち'})[reason]||reason;}
function renderDocuments(){const q=$('filter').value.trim().toLocaleLowerCase(),only=$('issues-only').checked;const items=state.documents.filter(d=>(!only||d.issues.length)&&`${d.title} ${d.issues.map(issueLabel).join(' ')}`.toLocaleLowerCase().includes(q));$('documents').replaceChildren();$('count').textContent=`${items.length}件`;
 for(const doc of items){const b=node('button');b.className='document';b.type='button';b.setAttribute('aria-current',String(doc.attachment_key===state.key));b.append(node('strong',doc.title),node('small',doc.issues.map(issueLabel).join(' · ')||'問題の記録なし'));b.addEventListener('click',()=>selectDocument(doc.attachment_key).catch(e=>notice(e.message)));$('documents').append(b);}}
async function selectDocument(key){if(state.dirty&&!confirm('未保存の修正があります。破棄して別の資料を開きますか？'))return;state.key=key;state.version=null;state.revision=0;state.edits={};state.toc=[];state.tocChanged=false;state.zones={};state.pageChunks=[];state.allAnchors=[];state.originals.clear();state.anchors.clear();state.dirty=false;state.reviewedRevision=null;$('history').replaceChildren();renderDocuments();toggleDocuments(false);pdfViewer.clear();pageCache.clear();$('paper-stage').hidden=true;$('history-detail').hidden=true;await loadPage(1,true);$('viewer').focus();await loadHistory();}
async function loadPage(page,initial=false){if(!state.key)return;if(!Number.isInteger(page)||page<1||!initial&&page>state.totalPages){notice('PDFページを文書の範囲内の整数で指定してください。');return;}const request=++state.request;notice('ページを読み込んでいます…');$('save').disabled=true;$('apply').disabled=true;try{const key=state.key;const [data]=await Promise.all([cachedPage(key,page),...(initial?[]:[api(`/admin/debug/api/documents/${key}/state?version=${encodeURIComponent(state.version)}`)])]);if(request!==state.request)return;
 if(state.version&&state.version!==data.version)throw new Error('原本または索引が変更されました。修正内容を保持して資料を開き直してください。');
 state.version=data.version;state.page=data.page;state.totalPages=data.total_pages;state.allAnchors=data.anchors||[];state.pageChunks=data.chunks;
 if(initial){state.revision=data.draft.revision;state.edits=data.draft_stale||data.draft_applied?{}:data.draft.edits;state.toc=data.draft_stale||data.draft_applied?[]:data.draft.toc;state.zones=data.draft_stale||data.draft_applied?{}:(data.draft.zones||{});state.tocChanged=!(data.draft_stale||data.draft_applied)&&Boolean(data.draft.toc_changed);if(data.draft_applied)notice('前回の修正はDBへ反映済みです。監査・検索公開の結果は処理状況で確認してください。');else if(data.draft_stale)notice('保存済み下書きは古い原本・索引のものです。適用せず、新しい原本で確認してください。');else notice('');}else notice('');
 state.originalToc=data.current_toc;$('title').textContent=data.title;$('page').value=data.page;$('page').max=data.total_pages;$('page-info').textContent=`/ ${data.total_pages}${data.page_label?` · 紙面 ${data.page_label}`:''}`;$('previous').disabled=data.page<=1;$('next').disabled=data.page>=data.total_pages;
 $('paper-stage').hidden=true;$('paper-placeholder').textContent='紙面を読み込んでいます…';$('paper-placeholder').hidden=false;
 const pdfUrl=`/admin/debug/api/documents/${state.key}/pdf?version=${encodeURIComponent(data.version)}`;$('source-pdf').href=pdfUrl;await pdfViewer.open(pdfUrl);if(request!==state.request)return;await fitPaper();if(request!==state.request)return;$('paper-stage').hidden=false;$('paper-placeholder').hidden=true;$('export').href=`/admin/debug/api/documents/${state.key}/draft/export`;
 for(const chunk of data.chunks){state.originals.set(chunk.id,chunk.text);state.anchors.set(chunk.id,data.page);}
 $('zone-start').value=data.page;$('zone-end').value=data.page;
 renderChunks();renderToc();renderNotePages();draftStatus();$('save').disabled=false;$('apply').disabled=false;prefetchPages();

 }catch(e){if(request===state.request){$('paper-stage').hidden=true;notice(`${e.message} 資料を開き直してください。`);}}}
const zoneNames={body:'本文',bibliography:'参考文献',endnote:'巻末注',footnote:'脚注',toc:'目次ページ',index:'索引',front_matter:'前付',back_matter:'後付',colophon:'奥付',other_paratext:'その他',corrupted:'認識崩れ'};
function chunkZone(chunk){return state.zones[chunk.id]??chunk.metadata.zone??'body';}
function renderChunks(){
 for(const [target,annotate] of [[$('chunks'),false],[$('notes-chunks'),true]]){
  target.replaceChildren();for(const chunk of state.pageChunks)renderChunk(chunk,target,annotate);
  if(!state.pageChunks.length)target.append(node('p','このPDFページに対応するチャンクはありません。白紙・未抽出・ページ情報を確認してください。'));
 }
}
function renderChunk(chunk,target=$('chunks'),annotate=false){
 const card=node('div');card.className='chunk';card.append(node('small',`${chunk.id} · ${zoneNames[chunkZone(chunk)]||chunkZone(chunk)}`));
 const original=node('details');original.append(node('summary','現在のDB本文'),node('div',chunk.text));card.append(original);
 const input=node('textarea');input.value=state.edits[chunk.id]??chunk.text;input.setAttribute('aria-label',`${chunk.id}の文字起こし修正`);
 input.addEventListener('input',()=>{if(input.value===chunk.text)delete state.edits[chunk.id];else state.edits[chunk.id]=input.value;markDirty();draftStatus();});card.append(input);
 const tools=node('div');tools.className='chunk-tools';
 if(annotate){const label=node('label','分類 '),select=node('select');select.setAttribute('aria-label',`${chunk.id}の分類`);for(const [value,name] of Object.entries(zoneNames)){const option=node('option',name);option.value=value;select.append(option);}select.value=chunkZone(chunk);select.addEventListener('change',()=>{if(select.value===(chunk.metadata.zone||'body'))delete state.zones[chunk.id];else state.zones[chunk.id]=select.value;markDirty();draftStatus();renderNotePages();});label.append(select);tools.append(label);}
 const heading=node('button','ここを節の開始位置にする');heading.type='button';heading.title='このチャンクの先頭から新しい節にします。Contentsページの指定ではありません。';heading.addEventListener('click',()=>addHeading(chunk));tools.append(heading);card.append(tools);target.append(card);
}
function initialiseToc(){if(!state.tocChanged&&!state.toc.length)state.toc=state.originalToc.map(x=>({...x}));state.tocChanged=true;}
function addHeading(chunk=null){
 const rows=tocRows();if(chunk&&rows.some(x=>x.chunk_id===chunk.id)){notice('この位置には既に見出しがあります。「目次・見出し構造」タブで編集してください。');setTab('toc');return;}
 const position=chunk?state.allAnchors.findIndex(x=>x.id===chunk.id):-1;
 let index=chunk?rows.findIndex(x=>x.chunk_id?state.allAnchors.findIndex(a=>a.id===x.chunk_id)>position:Number(x.page)>state.page):-1;
 insertHeading(index<0?rows.length:index,chunk);
}
function outlineRows(){return Array.from($('toc').children).filter(row=>row.className==='toc-row');}
function insertHeading(index,chunk=null){
 if(!state.key||!state.version){notice('先に資料を選択してください。');return;}
 const rows=tocRows();if(!Number.isInteger(index)||index<0||index>rows.length)return;
 const previous=rows[index-1],next=rows[index];
 const start=Math.max(1,Number(previous?.page)||1),end=Math.max(start,Number(next?.page)||state.totalPages);
 const page=chunk?state.page:Math.max(start,Math.min(end,state.page));
 const level=index===0?1:next?.level||previous?.level||1;
 initialiseToc();state.toc.splice(index,0,{...(chunk?{chunk_id:chunk.id}:{}),page,level,title:''});
 markDirty();setTab('toc');draftStatus();notice('見出し名を入力し、開始ページ・位置を確認してください。');
 outlineRows()[index]?.querySelector('input[aria-label="見出し名"]')?.focus();
}
function insertionPoint(index,depth){
 const gap=node('div');gap.className='toc-insert';gap.style.setProperty('--outline-depth',String(Math.max(0,depth-1)));
 const button=node('button','＋');button.type='button';button.setAttribute('aria-label',index===0?'先頭に見出しを追加':`見出し${index}の後に追加`);button.title='ここに見出しを追加';button.addEventListener('click',()=>insertHeading(index));gap.append(button);return gap;
}
function tocRows(){return state.tocChanged||state.toc.length?state.toc:state.originalToc;}
function moveHeading(index,delta){
 const selectionStart=document.activeElement?.selectionStart,selectionEnd=document.activeElement?.selectionEnd;
 const rows=tocRows(),entry=rows[index];if(!entry)return;
 const depth=entry.level+delta;let end=index+1;
 while(end<rows.length&&rows[end].level>entry.level)end++;
 if(depth<1||depth>8||delta>0&&(index===0||depth>rows[index-1].level+1)||rows.slice(index,end).some(x=>x.level+delta>8)){
  notice(delta>0?'この見出しはこれ以上下の階層にできません。先行する見出しと階層を確認してください。':'この見出しは最上位です。');return;
 }
 initialiseToc();for(let n=index;n<end;n++)state.toc[n].level+=delta;
 markDirty();renderToc();draftStatus();notice('');
 const input=outlineRows()[index]?.querySelector('input[aria-label="見出し名"]');input?.focus();if(Number.isInteger(selectionStart))input?.setSelectionRange?.(selectionStart,selectionEnd);
}
function renderToc(){
 const target=$('toc');target.replaceChildren();const entries=tocRows();
 for(const [index,entry] of entries.entries()){
  target.append(insertionPoint(index,entry.level));
  const row=node('div');row.className='toc-row';row.style.setProperty('--outline-depth',String(Math.max(0,entry.level-1)));row.setAttribute('aria-label',`${entry.level}階層目の見出し`);
  function edit(){initialiseToc();return state.toc[index];}
  const controls=node('div');controls.className='outline-controls';
  const outdent=node('button','←'),indent=node('button','→');outdent.type=indent.type='button';
  outdent.setAttribute('aria-label','階層を上げる');indent.setAttribute('aria-label','階層を下げる');
  outdent.title='階層を上げる（Shift+Tab）';indent.title='階層を下げる（Tab）';outdent.disabled=entry.level<=1;indent.disabled=index===0||entry.level>=8||entry.level>tocRows()[index-1].level;
  outdent.addEventListener('click',()=>moveHeading(index,-1));indent.addEventListener('click',()=>moveHeading(index,1));controls.append(outdent,indent);
  const titleLabel=node('label'),title=node('input');title.value=entry.title;title.setAttribute('aria-label','見出し名');title.addEventListener('input',()=>{edit().title=title.value;markDirty();draftStatus();});titleLabel.append(title);title.addEventListener('keydown',event=>{if(event.isComposing||event.ctrlKey||event.metaKey||event.altKey)return;if(event.key==='Tab'){event.preventDefault();moveHeading(index,event.shiftKey?-1:1);}else if(event.key==='Escape'){event.preventDefault();locationDetails.open=true;page.focus();}});
  const location=node('div');location.className='toc-location';const pageLabel=node('label','開始PDF頁'),page=node('input');page.type='text';page.inputMode='numeric';page.min='1';page.max=state.totalPages;page.value=entry.page||state.anchors.get(entry.chunk_id)||'';page.setAttribute('aria-label','見出し開始PDFページ');
  page.addEventListener('change',()=>{const value=edit();value.page=Number(page.value);delete value.chunk_id;markDirty();renderToc();draftStatus();});pageLabel.append(page);
  const anchorLabel=node('label','ページ内の開始位置'),anchor=node('select');anchor.setAttribute('aria-label','見出しのページ内開始位置');const first=node('option','ページの最初');first.value='';anchor.append(first);
  const candidates=state.allAnchors.filter(x=>Number(x.page)===Number(page.value));
  for(const [n,candidate] of candidates.entries()){const option=node('option',`位置 ${n+1}：${candidate.preview||'（本文なし）'}`);option.value=candidate.id;anchor.append(option);}
  const preview=node('p');preview.className='anchor-preview';preview.setAttribute('aria-live','polite');
  function showAnchorPreview(){const selected=candidates.find(x=>x.id===anchor.value);const start=selected||(!anchor.value&&candidates[0]);preview.textContent=start?`書き出し：${start.preview||'（本文なし）'}`:'この位置の本文を確認できません。';}
  anchor.value=entry.chunk_id||'';showAnchorPreview();
  anchor.addEventListener('change',()=>{const value=edit();if(anchor.value)value.chunk_id=anchor.value;else delete value.chunk_id;showAnchorPreview();markDirty();draftStatus();});anchorLabel.append(anchor,preview);
  const go=node('button','紙面へ');go.type='button';go.addEventListener('click',()=>loadPage(Number(page.value)));
  const remove=node('button','削除');remove.type='button';remove.addEventListener('click',()=>{const value=edit();state.toc=state.toc.filter(x=>x!==value);markDirty();renderToc();draftStatus();});
  location.append(pageLabel,anchorLabel,go,remove);const locationDetails=node('details');locationDetails.className='outline-location';locationDetails.append(node('summary',`PDF ${page.value||'?'}頁 · 開始位置を編集`),location);row.append(controls,titleLabel,locationDetails);
  if(!candidates.length&&!entry.chunk_id)row.append(Object.assign(node('p','このページは未抽出です。下書き保存はできますが、反映前に開始位置の確認が必要です。'),{className:'unresolved'}));target.append(row);
 }
 target.append(insertionPoint(entries.length,entries.at(-1)?.level||1));
 if(!entries.length)target.append(node('p','この上にカーソルを合わせると「＋」が表示されます。クリックして最初の見出しを追加できます。'));
}
function orderedToc(){return [...state.toc];}
async function save(){if(!state.key||!state.version)return;const key=state.key,version=state.version,generation=state.generation;$('save').disabled=true;try{const saved=await api(`/admin/debug/api/documents/${state.key}/draft`,{method:'POST',body:JSON.stringify({revision:state.revision,version:state.version,edits:state.edits,toc:orderedToc(),toc_changed:state.tocChanged,zones:state.zones})});if(key!==state.key||version!==state.version )return;state.revision=saved.revision;state.dirty=state.generation!==generation;state.reviewedRevision=null;draftStatus();$('history').replaceChildren();await loadHistory();notice(state.dirty?'保存中に追加された編集は未保存です。もう一度保存してください。':'下書きを保存しました。検索DBにはまだ反映していません。');}catch(e){notice(e.message);}finally{if(key===state.key&&version===state.version)$('save').disabled=false;}}
function localDifferences(){const target=$('diff-content');target.replaceChildren();for(const [id,text] of Object.entries(state.edits)){target.append(node('h4',id));const pair=node('div');pair.className='diff-pair';pair.append(node('pre',state.originals.get(id)??'このページ以外の原文は、該当ページで確認してください。'),node('pre',text));target.append(pair);}for(const [id,zone] of Object.entries(state.zones))target.append(node('p',`${id}：分類を${zoneNames[zone]||zone}へ変更`));target.append(node('h4','目次の修正案'),node('pre',orderedToc().map(x=>`${'　'.repeat(Math.max(0,x.level-1))}${x.title}（PDF ${x.page||'?'}頁）`).join('\n')||'変更なし'));setTab('diff');}
$('filter').addEventListener('input',renderDocuments);$('issues-only').addEventListener('change',renderDocuments);$('previous').addEventListener('click',()=>loadPage(state.page-1));$('next').addEventListener('click',()=>loadPage(state.page+1));$('page').addEventListener('change',()=>loadPage(Number($('page').value)));$('save').addEventListener('click',save);$('diff').addEventListener('click',()=>differences().catch(e=>notice(e.message)));window.addEventListener('beforeunload',event=>{if(state.dirty){event.preventDefault();event.returnValue='';}});
api('/admin/debug/api/documents').then(items=>{state.documents=items;renderDocuments();}).catch(e=>notice(e.message));

$('apply').addEventListener('click',async()=>{
 if(!Object.keys(state.edits).length&&!state.toc.length&&!state.tocChanged&&!Object.keys(state.zones).length){notice('反映する修正がありません。本文・目次・分類を編集してください。');return;}
 if(state.dirty||state.revision<1){notice('先に下書きを保存し、差分を確認してください。');return;}
 if(state.reviewedRevision!==state.revision){try{await differences();notice('保存済みの差分を確認後、もう一度DBへ反映を押してください。');}catch(e){notice(e.message);}return;}
 if(!confirm('保存済みの本文・目次修正を検索DBへ反映します。ローカルの再埋め込みと全監査を行います。原本PDFは変更しません。続けますか？'))return;
 $('apply').disabled=true;
 try{const record=await api(`/admin/debug/api/documents/${state.key}/apply`,{method:'POST',body:JSON.stringify({revision:state.revision,version:state.version})});notice(`修正反映を開始しました（${record.id}）。処理状況ページで進捗・監査結果を確認できます。`);}
 catch(e){notice(e.message);}
 finally{$('apply').disabled=false;}
});

async function differences(){
 if(state.dirty||!state.revision){localDifferences();return;}
 const key=state.key,revision=state.revision;const data=await api(`/admin/debug/api/documents/${key}/diff?revision=${revision}`);if(key!==state.key||revision!==state.revision||state.dirty)return;
 const target=$('diff-content');target.replaceChildren();
 for(const change of data.text_changes){target.append(node('h4',change.chunk_id));const pair=node('div');pair.className='diff-pair';pair.append(node('pre',change.before),node('pre',change.after));target.append(pair);}
 for(const change of data.zone_changes||[])target.append(node('p',`${change.chunk_id}：${zoneNames[change.before]||change.before} → ${zoneNames[change.after]||change.after}`));
 const pair=node('div');pair.className='diff-pair';
 const format=rows=>rows.map(x=>`${'　'.repeat(Math.max(0,x.level-1))}${x.title}（PDF ${x.page||'?'}頁）`).join('\n')||'変更なし';
 pair.append(node('pre',format(data.toc_before)),node('pre',format(data.toc_after)));target.append(node('h4','目次・見出し構造：現在 → 修正案'),pair);setTab('diff');state.reviewedRevision=data.revision;
}
$('copy-toc').addEventListener('click',()=>{if(!state.key||!state.version){notice('先に資料を選択してください。');return;}if(state.toc.length&&!confirm('編集中の見出し構造を現在のDBの見出し構造で置き換えますか？'))return;state.tocChanged=true;state.toc=state.originalToc.map(x=>({...x}));markDirty();renderToc();draftStatus();});

let historyBefore=null;
async function loadHistory(before=null){
 if(!state.key)return;
 const key=state.key;const rows=await api(`/admin/debug/api/documents/${key}/history${before?`?before=${before}`:''}`);
 if(key!==state.key)return;
 for(const row of rows){const card=node('p',`第${row.revision}版 · ${row.created_at} · ${row.actor} UTC · 本文 ${row.text_changes}件 / 見出し ${row.headings}件 / 分類 ${row.zone_changes||0}件 `);
 const show=node('button','内容');show.addEventListener('click',async()=>{try{const value=await api(`/admin/debug/api/documents/${key}/history/${row.revision}`);if(key!==state.key)return;$('history-detail').textContent=JSON.stringify(value,null,2);$('history-detail').hidden=false;}catch(e){notice(e.message);}});
 const undo=node('button','この修正前に戻す案');undo.addEventListener('click',()=>restoreHistory(key,row.revision,true).catch(e=>notice(e.message)));
 const redo=node('button','この修正案を復元');redo.addEventListener('click',()=>restoreHistory(key,row.revision,false).catch(e=>notice(e.message)));
 card.append(show,undo,redo);$('history').append(card);}
 historyBefore=rows.length?rows.at(-1).revision:null;$('older-history').disabled=rows.length<50;
}
async function restoreHistory(key,revision,before){
 if(key!==state.key)return;
 if(!confirm('編集中の下書きを、この版の修正箇所で置き換えます。別の箇所への修正は自動で取り消しません。新しい版として保存しますか？'))return;
 const saved=await api(`/admin/debug/api/documents/${key}/history/${revision}/restore?before=${before}`,{method:'POST',body:JSON.stringify({revision:state.revision,version:state.version})});
 if(key!==state.key)return;
 state.revision=saved.revision;state.edits=saved.edits;state.zones=saved.zones||{};state.toc=saved.toc;state.tocChanged=Boolean(saved.toc_changed);state.reviewedRevision=null;state.dirty=false;$('history').replaceChildren();pageCache.clear();await loadPage(state.page);await loadHistory();notice('履歴から新しい下書きを作成しました。差分を確認してからDBへ反映してください。');
}
$('older-history').addEventListener('click',()=>loadHistory(historyBefore).catch(e=>notice(e.message)));

function toggleDocuments(open){$('document-panel').hidden=!open;if(open)$('filter').focus();$('toggle-documents').setAttribute('aria-expanded',String(open));}
$('toggle-documents').addEventListener('click',()=>toggleDocuments($('document-panel').hidden));$('close-documents').addEventListener('click',()=>toggleDocuments(false));
const tabNames=['body','toc','notes','diff','history'];
function setTab(name){state.tab=name;for(const tab of tabNames){$('panel-'+tab).hidden=tab!==name;$('tab-'+tab).setAttribute('aria-selected',String(tab===name));$('tab-'+tab).tabIndex=tab===name?0:-1;}if(name==='body'||name==='notes')renderChunks();if(name==='toc')renderToc();}
for(const [n,tab] of tabNames.entries()){$('tab-'+tab).addEventListener('click',()=>{setTab(tab);if(tab==='diff')differences().catch(e=>notice(e.message));});$('tab-'+tab).addEventListener('keydown',event=>{if(event.key!=='ArrowLeft'&&event.key!=='ArrowRight')return;event.preventDefault();const next=tabNames[(n+(event.key==='ArrowRight'?1:tabNames.length-1))%tabNames.length];setTab(next);$('tab-'+next).focus();if(next==='diff')differences().catch(e=>notice(e.message));});}

function renderNotePages(){const pages=[...new Set(state.allAnchors.filter(x=>['footnote','endnote','bibliography'].includes(state.zones[x.id]??x.zone)).map(x=>Number(x.page)).filter(x=>x>0))].sort((a,b)=>a-b);$('note-pages').replaceChildren();for(const page of pages){const go=node('button',`PDF ${page}頁`);go.addEventListener('click',()=>loadPage(page));$('note-pages').append(go);}}
$('assign-zone').addEventListener('click',()=>{const start=Number($('zone-start').value),end=Number($('zone-end').value),zone=$('zone-kind').value;if(!Number.isInteger(start)||!Number.isInteger(end)||start<1||end<start||end>state.totalPages){notice('開始・終了PDFページを文書の範囲内で指定してください。');return;}const anchors=state.allAnchors.filter(x=>Number(x.page)>=start&&Number(x.page)<=end);const zones={...state.zones};for(const anchor of anchors){if(zone===anchor.zone)delete zones[anchor.id];else zones[anchor.id]=zone;}if(Object.keys(zones).length>500){notice('分類変更は一度に500チャンクまでです。ページ範囲を分けてください。');return;}if(!anchors.length){notice('指定範囲にチャンクがありません。未抽出のページは分類を反映できません。');return;}state.zones=zones;markDirty();draftStatus();renderChunks();renderNotePages();notice(`${anchors.length}チャンクの分類を修正案へ追加しました。`);});
async function fitPaper(){if(!pdfViewer.pdf)return;const requested=Number($('zoom').value);if($('zoom-mode').value==='custom')$('zoom').value=Number.isFinite(requested)&&requested>0?Math.max(5,Math.min(400,requested)):100;const percent=await pdfViewer.show(state.page,$('zoom-mode').value,$('zoom').value);if(percent!==null&&$('zoom-mode').value!=='custom')$('zoom').value=percent;}
function redrawPaper(){fitPaper().then(()=>pdfViewer.prefetch(state.page,$('zoom-mode').value,$('zoom').value)).catch(error=>notice(`PDFを表示できません。${error.message} 原本PDFリンクでも確認できます。`));}
$('zoom-mode').addEventListener('change',redrawPaper);$('zoom').addEventListener('change',()=>{$('zoom-mode').value='custom';redrawPaper();});
function changeZoom(delta){$('zoom-mode').value='custom';$('zoom').value=Math.max(5,Math.min(400,(Number($('zoom').value)||100)+delta));redrawPaper();}
$('zoom-out').addEventListener('click',()=>changeZoom(-25));$('zoom-in').addEventListener('click',()=>changeZoom(25));
if(typeof ResizeObserver!=='undefined')new ResizeObserver(redrawPaper).observe($('paper-viewport'));
window.addEventListener('keydown',event=>{if(event.key==='Escape'){toggleDocuments(false);return;}if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||!state.key)return;const target=event.target;if(target?.isContentEditable||['INPUT','TEXTAREA','SELECT','BUTTON'].includes(target?.tagName))return;if(event.key==='ArrowRight'&&state.page<state.totalPages){event.preventDefault();loadPage(state.page+1);}if(event.key==='ArrowLeft'&&state.page>1){event.preventDefault();loadPage(state.page-1);}});
