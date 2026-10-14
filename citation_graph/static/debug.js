'use strict';
const $=id=>document.getElementById(id);
const state={documents:[],key:null,page:1,version:null,revision:0,edits:{},toc:[],tocChanged:false,originals:new Map(),anchors:new Map(),dirty:false,generation:0,request:0,originalToc:[],reviewedRevision:null};
function node(tag,text=''){const e=document.createElement(tag);e.textContent=text;return e;}
function notice(text){$('notice').textContent=text;}
function markDirty(){state.dirty=true;state.generation++;state.reviewedRevision=null;}
async function api(path,options={}){const response=await fetch(path,{credentials:'same-origin',headers:{'Content-Type':'application/json'},...options});if(!response.ok){let text=`HTTP ${response.status}`;try{text=(await response.json()).error||text;}catch{}throw new Error(text);}return response.json();}
function draftStatus(){ $('draft-state').textContent=`下書き 第${state.revision}版 · 本文修正 ${Object.keys(state.edits).length}件 · 見出し ${state.toc.length}件${state.dirty?' · 未保存':''}`;}
function issueLabel(reason){return ({flat_fallback:'目次の階層を復元できません',failed:'抽出・構造処理の失敗',blocked:'処理を保留中',degraded:'抽出・構造の要確認',awaiting_mistral_ocr_batch:'OCR待ち'})[reason]||reason;}
function renderDocuments(){const q=$('filter').value.trim().toLocaleLowerCase(),only=$('issues-only').checked;const items=state.documents.filter(d=>(!only||d.issues.length)&&`${d.title} ${d.issues.map(issueLabel).join(' ')}`.toLocaleLowerCase().includes(q));$('documents').replaceChildren();$('count').textContent=`${items.length}件`;
 for(const doc of items){const b=node('button');b.className='document';b.type='button';b.setAttribute('aria-current',String(doc.attachment_key===state.key));b.append(node('strong',doc.title),node('small',doc.issues.map(issueLabel).join(' · ')||'問題の記録なし'));b.addEventListener('click',()=>selectDocument(doc.attachment_key).catch(e=>notice(e.message)));$('documents').append(b);}}
async function selectDocument(key){if(state.dirty&&!confirm('未保存の修正があります。破棄して別の資料を開きますか？'))return;state.key=key;state.version=null;state.revision=0;state.edits={};state.toc=[];state.tocChanged=false;state.originals.clear();state.anchors.clear();state.dirty=false;state.reviewedRevision=null;$('history').replaceChildren();renderDocuments();await loadPage(1,true);await loadHistory();}
async function loadPage(page,initial=false){if(!state.key)return;const request=++state.request;notice('ページを読み込んでいます…');$('save').disabled=true;try{const data=await api(`/admin/debug/api/documents/${state.key}/pages/${page}`);if(request!==state.request)return;
 if(state.version&&state.version!==data.version){notice('原本または索引が変更されました。修正を保存せず、資料を開き直してください。');return;}
 state.version=data.version;state.page=data.page;
 if(initial){state.revision=data.draft.revision;state.edits=data.draft_stale||data.draft_applied?{}:data.draft.edits;state.toc=data.draft_stale||data.draft_applied?[]:data.draft.toc;state.tocChanged=!(data.draft_stale||data.draft_applied)&&Boolean(data.draft.toc_changed);if(data.draft_applied)notice('前回の修正はDBへ反映済みです。監査・検索公開の結果は処理状況で確認してください。');else if(data.draft_stale)notice('保存済み下書きは古い原本・索引のものです。適用せず、新しい原本で確認してください。');else notice('');}else notice('');
 state.originalToc=data.current_toc;$('current-toc').replaceChildren(...data.current_toc.map(x=>node('p',`${'　'.repeat(Math.max(0,x.level-1))}${x.title}（PDF ${x.page||'?'}頁）`)));$('tools').hidden=false;$('title').textContent=data.title;$('page').value=data.page;$('page').max=data.total_pages;$('page-info').textContent=`/ ${data.total_pages}ページ${data.page_label?` · 紙面番号 ${data.page_label}`:''}`;$('previous').disabled=data.page<=1;$('next').disabled=data.page>=data.total_pages;
 $('paper').src=`/admin/debug/api/documents/${state.key}/pages/${data.page}/image?version=${encodeURIComponent(data.version)}`;$('export').href=`/admin/debug/api/documents/${state.key}/draft/export`;
 $('chunks').replaceChildren();for(const chunk of data.chunks){state.originals.set(chunk.id,chunk.text);state.anchors.set(chunk.id,data.page);renderChunk(chunk);}if(!data.chunks.length)$('chunks').append(node('p','この物理PDFページに対応するチャンクはありません。白紙・未抽出・ページ情報の欠落などを確認してください。'));renderToc();draftStatus();$('save').disabled=false;
 }catch(e){if(request===state.request)notice(e.message);}}
function renderChunk(chunk){const card=node('div');card.className='chunk';card.append(node('small',`${chunk.id} · ${chunk.metadata.block_type||''} · ${chunk.metadata.zone||''}`));const original=node('details'),summary=node('summary','抽出時の本文を確認');original.append(summary,node('div',chunk.text));card.append(original);const input=node('textarea');input.value=state.edits[chunk.id]??chunk.text;input.setAttribute('aria-label',`${chunk.id}の文字起こし修正`);input.addEventListener('input',()=>{if(input.value===chunk.text)delete state.edits[chunk.id];else state.edits[chunk.id]=input.value;markDirty();draftStatus();});card.append(input);
 const heading=node('button','ここから見出し');heading.type='button';heading.addEventListener('click',()=>{if(state.toc.some(x=>x.chunk_id===chunk.id)){notice('このチャンクには既に見出しがあります。下の目次欄で編集してください。');return;}state.tocChanged=true;state.toc.push({chunk_id:chunk.id,page:state.page,level:1,title:''});markDirty();renderToc();draftStatus();});card.append(heading);$('chunks').append(card);}
function renderToc(){const target=$('toc');target.replaceChildren();for(const entry of state.toc){const row=node('div');row.className='toc-row';row.append(node('small',`PDF ${entry.page||state.anchors.get(entry.chunk_id)||'?'}頁`));const level=node('input');level.type='number';level.min='1';level.max='8';level.value=entry.level;level.setAttribute('aria-label','見出し階層');level.addEventListener('input',()=>{state.tocChanged=true;entry.level=Number(level.value);markDirty();draftStatus();});const title=node('input');title.value=entry.title;title.placeholder='見出し名';title.setAttribute('aria-label','見出し名');title.addEventListener('input',()=>{state.tocChanged=true;entry.title=title.value;markDirty();draftStatus();});const remove=node('button','削除');remove.type='button';remove.addEventListener('click',()=>{state.tocChanged=true;state.toc=state.toc.filter(x=>x!==entry);markDirty();renderToc();draftStatus();});row.append(level,title,remove);target.append(row);}if(!state.toc.length)target.append(node('p','目次の修正案はまだありません。'));}
function orderedToc(){return [...state.toc].sort((a,b)=>(a.page||0)-(b.page||0)||a.chunk_id.localeCompare(b.chunk_id,undefined,{numeric:true}));}
async function save(){if(!state.key||!state.version)return;const key=state.key,version=state.version,request=state.request,generation=state.generation;$('save').disabled=true;try{const saved=await api(`/admin/debug/api/documents/${state.key}/draft`,{method:'POST',body:JSON.stringify({revision:state.revision,version:state.version,edits:state.edits,toc:orderedToc(),toc_changed:state.tocChanged})});if(key!==state.key||version!==state.version||request!==state.request)return;state.revision=saved.revision;state.dirty=state.generation!==generation;state.reviewedRevision=null;draftStatus();$('history').replaceChildren();await loadHistory();notice(state.dirty?'保存中に追加された編集は未保存です。もう一度保存してください。':'下書きを保存しました。検索DBにはまだ反映していません。');}catch(e){notice(e.message);}finally{if(key===state.key&&request===state.request)$('save').disabled=false;}}
function localDifferences(){const target=$('diff-content');target.replaceChildren();for(const [id,text] of Object.entries(state.edits)){target.append(node('h4',id));const pair=node('div');pair.className='diff-pair';pair.append(node('pre',state.originals.get(id)??'このページ以外の原文は、該当ページで確認してください。'),node('pre',text));target.append(pair);}target.append(node('h4','目次の修正案'),node('pre',orderedToc().map(x=>`${'　'.repeat(Math.max(0,x.level-1))}${x.title}（PDF ${x.page||'?'}頁）`).join('\n')||'変更なし'));$('differences').hidden=false;}
$('filter').addEventListener('input',renderDocuments);$('issues-only').addEventListener('change',renderDocuments);$('previous').addEventListener('click',()=>loadPage(state.page-1));$('next').addEventListener('click',()=>loadPage(state.page+1));$('page').addEventListener('change',()=>loadPage(Number($('page').value)));$('save').addEventListener('click',save);$('diff').addEventListener('click',()=>differences().catch(e=>notice(e.message)));$('paper').addEventListener('error',()=>notice('紙面画像を読み込めません。原本・索引の変更や認証状態を確認し、ページを再読込してください。'));window.addEventListener('beforeunload',event=>{if(state.dirty){event.preventDefault();event.returnValue='';}});
api('/admin/debug/api/documents').then(items=>{state.documents=items;renderDocuments();}).catch(e=>notice(e.message));

$('apply').addEventListener('click',async()=>{
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
 const pair=node('div');pair.className='diff-pair';
 const format=rows=>rows.map(x=>`${'　'.repeat(Math.max(0,x.level-1))}${x.title}`).join('\n')||'変更なし';
 pair.append(node('pre',format(data.toc_before)),node('pre',format(data.toc_after)));target.append(node('h4','目次：現在 → 修正案'),pair);$('differences').hidden=false;state.reviewedRevision=data.revision;
}
$('copy-toc').addEventListener('click',()=>{if(state.toc.length&&!confirm('編集中の目次を現在の目次で置き換えますか？'))return;state.tocChanged=true;state.toc=state.originalToc.map(x=>({...x}));markDirty();renderToc();draftStatus();});

let historyBefore=null;
async function loadHistory(before=null){
 if(!state.key)return;
 const key=state.key;const rows=await api(`/admin/debug/api/documents/${key}/history${before?`?before=${before}`:''}`);
 if(key!==state.key)return;
 for(const row of rows){const card=node('p',`第${row.revision}版 · ${row.created_at} · ${row.actor} · 本文 ${row.text_changes}件 / 見出し ${row.headings}件 `);
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
 state.revision=saved.revision;state.edits=saved.edits;state.toc=saved.toc;state.tocChanged=Boolean(saved.toc_changed);state.reviewedRevision=null;state.dirty=false;$('history').replaceChildren();await loadPage(state.page);await loadHistory();notice('履歴から新しい下書きを作成しました。差分を確認してからDBへ反映してください。');
}
$('older-history').addEventListener('click',()=>loadHistory(historyBefore).catch(e=>notice(e.message)));
