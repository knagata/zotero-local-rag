"use strict";
const $ = (id) => document.getElementById(id);
let state = null;
let selectedDefinition = null;
let historySignature = "";
let selectedStages = null;
let actionSignature = "";

function text(value) { return value === null || value === undefined ? "—" : String(value); }
function localTime(value) { return value ? new Date(value).toLocaleString("ja-JP") : "—"; }
function statusLabel(value) {
  return {completed:"完了",running:"処理中",in_progress:"処理中",submitted:"送信済み",queued:"開始待ち",success:"処理完了",collected:"回収済み",stopping:"停止中",failed:"失敗",cancelled:"停止済み",rejected:"拒否"}[value] || value;
}
function node(tag, className, content) {
  const element = document.createElement(tag); if (className) element.className = className;
  if (content !== undefined) element.textContent = content; return element;
}
async function api(path, options={}) {
  const response = await fetch(path, {credentials:"same-origin", headers:{"Content-Type":"application/json"}, ...options});
  if (response.status === 401) { location.href = "/auth/login?next=%2Fadmin%2F"; throw new Error("認証が必要です"); }
  const payload = response.headers.get("content-type")?.includes("json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`); return payload;
}
function metric(label, value, detail="") {
  const box=node("div","metric"); box.append(node("small","",label),node("b","",text(value))); if(detail) box.append(node("small","",detail)); return box;
}
function renderMetrics(data) {
  const m=data.manifest, a=data.artifacts, gate=data.database_gate, batch=data.mistral_batch; const target=$("metrics"); target.replaceChildren(
    metric("索引済み添付",m.attachments,`ノート ${m.notes}`), metric("HNSW検証",m.hnsw_validated?"正常":"要確認",`inflight ${m.inflight}`),
    metric("DB監査gate",gate.passed?"合格":gate.available?"不合格":"結果なし",localTime(gate.modified_at)), metric("未解決artifact",a.unresolved ?? "—",a.error || "failed + blocked"),
    metric("索引書き込み",data.indexing_lock?"実行中":"停止中",data.indexing_lock?.started_at || "ロックなし"),
    metric("検索用の添付",data.search_snapshot?.available ? data.search_snapshot.attachments : "未分離",data.search_snapshot?.available ? `更新中も検索可能 · ${localTime(data.search_snapshot.published_at)}時点（新しい資料は監査後に反映）` : "検索と更新は同じDBを使用"),
    metric("Mistral Batch",batch?statusLabel(batch.phase):"未実行",batch?`${batch.completed_requests}/${batch.total_requests || batch.candidate_count}件 · 確認 ${localTime(batch.last_checked_at)}`:"")
  ); $("updated").textContent=`更新 ${new Date().toLocaleTimeString("ja-JP")}`;
  const diagnostics=$("audit-diagnostics"); diagnostics.replaceChildren();
  if(gate.audit_status) {
    diagnostics.append(node("p","freshness-warning",`${gate.audit_status.message}（${localTime(gate.audit_status.updated_at)}）`));
    if(gate.audit_status.status==="failed") for(const finding of gate.audit_status.findings||[]) {
      diagnostics.append(node("p","freshness-warning",`${finding.title||""} ${finding.item_key||finding.attachment_key||""}：${finding.message} ${finding.action}`));
    }
    if(gate.audit_status.repair?.backup) diagnostics.append(node("p","",`修復前バックアップ：${gate.audit_status.repair.backup}`));
  }
  if(!gate.passed && gate.available) {
    diagnostics.append(node("p","freshness-warning",`直近のDB監査：資料単位の不合格 ${gate.failed_items||0}件（${localTime(gate.modified_at)}）。更新状況の確認とは別の検査です。`));
    for(const failure of gate.failures||[]) {
      diagnostics.append(node("p","freshness-warning",`『${failure.title}』（${failure.item_key}）：${failure.message} ${failure.action}`));
    }
    for(const detail of gate.global_details||[]) diagnostics.append(node("p","freshness-warning",`${detail.message} ${detail.action}`));
  }
}
function freshnessCard(label, value, detail, keys=[], attention=0) {
  const count=Number.parseInt(value,10), state=attention>0?"attention":count>0?"pending":"current";
  const box=node("div",`freshness-card ${state}`);
  box.append(node("span","",label),node("b",state,value),node("small","",detail));
  if(keys.length) box.append(node("code","",keys.join(" · "))); return box;
}
function structureFreshnessDetail(structure) {
  if(structure.flat_fallback !== undefined) return `未処理 ${structure.unprocessed||0}件・更新待ち ${structure.outdated||0}件・処理失敗 ${structure.failed||0}件・その他要確認 ${structure.degraded||0}件・章立て未復元 ${structure.flat_fallback}件（本文検索可）`;
  return structure.attention?`失敗・要確認 ${structure.attention}件`:structure.lightweight?"簡易確認（保存された処理状態）":"保存チャンクと照合";
}
function renderFreshness(data) {
  const target=$("freshness"), report=data.update_status; target.replaceChildren();
  if(!report) {
    target.append(node("p","freshness-empty","まだ確認していません。「更新状況を確認」を実行してください。"));
    $("freshness-updated").textContent="未確認"; return;
  }
  const index=report.index||{}, structure=report.structure||{}, summaries=report.summaries||{}, citations=report.citations||{};
  if(report.check_failed) target.append(node("p","freshness-warning",`最新の確認は失敗しました（${localTime(report.check_failed_at)}）。表示値は前回成功時点です。`));
  else if(report.recheck_pending) target.append(node("p","freshness-warning","更新処理後の再確認を待っています。表示値は前回確認時点です。"));
  else if(report.stale) target.append(node("p","freshness-warning","確認結果が古くなっています。定期確認の状態を確認してください。"));
  target.append(
    freshnessCard("索引",`${text(index.pending)}件`,index.attention?`原本欠落 ${index.attention}件`:index.pending?`添付・ノートの差分（ノート ${index.notes?.pending||0}件）`:"差分なし",index.sample_keys||[],index.attention||0),
    freshnessCard("文書構造・目次",`${text(structure.pending)}件`,structureFreshnessDetail(structure),structure.sample_keys||[],structure.attention||0),
    freshnessCard("階層要約",`${text(summaries.pending)}件`,summaries.attention?`要確認 ${summaries.attention}項目・索引不整合 ${summaries.index_issues||0}件`:summaries.lightweight?"簡易確認（要約索引の整合性は未検査）":`要約索引 ${summaries.actual_index_rows||0}件・整合`,summaries.sample_keys||[],summaries.attention||0),
    freshnessCard("Citation Network",`${text(citations.pending)}件`,citations.attention?`うちエラー ${citations.attention}件`:citations.metadata_changed?`書誌情報変更 ${citations.metadata_changed}件`:"未処理・再試行対象",citations.sample_keys||[],citations.attention||0)
  );
  $("freshness-updated").textContent=report.recheck_pending?"再確認待ち":report.stale?"期限切れ":`確認 ${localTime(report.generated_at)}`;
}
function renderActions(data) {
  if(!Array.isArray(data.stages)||!data.stages.length) {
    actionSignature="";
    $("actions").replaceChildren(node("p","freshness-warning","工程一覧を取得できません。管理画面サービスが旧版のまま動いている可能性があります。Macのターミナルで次を実行し、この画面を再読み込みしてください。"),node("pre","",".venv/bin/python scripts/manage_citation_graph_service.py restart"));
    return;
  }
  const stages=data.stages, busy=Boolean(data.active_job);
  if(selectedStages===null) selectedStages=new Set(stages.filter(x=>x.default).map(x=>x.key));
  for(const stage of stages) if(stage.requires.some(key=>!selectedStages.has(key))) selectedStages.delete(stage.key);
  const signature=JSON.stringify([stages,busy,[...selectedStages]]);
  if(signature===actionSignature)return; actionSignature=signature;
  const target=$("actions"); target.replaceChildren();
  stages.forEach(stage=>{
    const missing=stage.requires.filter(key=>!selectedStages.has(key));
    const label=node("label",`action stage${missing.length?" unavailable":""}`), input=node("input");
    input.type="checkbox"; input.checked=selectedStages.has(stage.key); input.disabled=busy||missing.length>0;
    input.addEventListener("change",()=>{if(input.checked)selectedStages.add(stage.key);else selectedStages.delete(stage.key);renderActions(state);});
    const detail=node("span"); detail.append(node("strong","",stage.label),node("p","",stage.description));
    if(stage.paid)detail.append(node("small","paid","設定・処理内容によりAPI課金あり"));
    if(missing.length)detail.append(node("small","",`前工程を選択してください：${stages.filter(x=>missing.includes(x.key)).map(x=>x.label).join("、")}`));
    label.append(input,detail);target.append(label);
  });
  const run=node("button","","選択した工程を実行");run.disabled=busy||!selectedStages.size;
  run.addEventListener("click",()=>openConfirmation({key:"selected_steps",label:"選択した工程",selected_steps:stages.filter(x=>selectedStages.has(x.key)).map(x=>x.key),paid:stages.some(x=>selectedStages.has(x.key)&&x.paid),description:stages.filter(x=>selectedStages.has(x.key)).map(x=>x.label).join(" → ")}));
  const reset=node("button","secondary","クイック実行の選択に戻す"); reset.disabled=busy;
  reset.addEventListener("click",()=>{selectedStages=null;renderActions(state);});target.append(run,reset);
}
function logProgress(log) {
  const matches=[...String(log||"").matchAll(/(?:PROGRESS|進捗)[^\n]*?(\d+)\s*\/\s*(\d+)/gi)];
  for(let index=matches.length-1;index>=0;index-=1){const current=Number(matches[index][1]),total=Number(matches[index][2]);if(total>0&&current>=0&&current<=total)return current/total;}
  return null;
}
function progressEstimate(job,log="") {
  if(job.status==="completed") return {percent:100,label:"100% · 完了",estimated:false};
  const total=Math.max(1,Number(job.step_total)||1),step=Math.max(0,Number(job.step_index)||0);
  if(job.status==="queued") return {percent:2,label:`開始待ち · 0/${total}ステップ`,estimated:true};
  const measured=logProgress(log),started=job.started_at?new Date(job.started_at).getTime():Date.now();
  const elapsed=Math.max(0,(Date.now()-started)/1000);
  const within=measured??Math.min(.9,.1+elapsed/(elapsed+300)*.8);
  const percent=Math.max(2,Math.min(99,((Math.max(1,step)-1+within)/total)*100));
  const detail=measured===null?"経過時間からの目安":`${Math.round(measured*100)}% · ログ実測`;
  return {percent,label:`${Math.round(percent)}% · ${step||1}/${total}ステップ · ${detail}`,estimated:measured===null};
}
function jobElement(job,progress=null) {
  const row=node("div","job"); const left=node("div"); left.append(node("strong","",job.label),node("p","",job.current_step || job.type));
  const right=node("div","job-meta"); right.append(node("span",`status ${job.status}`,statusLabel(job.status)),node("time","",localTime(job.started_at || job.created_at))); row.append(left,right);
  if(progress){const wrap=node("div","job-progress");const labels=node("div","job-progress-labels");labels.append(node("span","","進捗"),node("span","",progress.label));const track=node("div",`progress-track${progress.estimated?" estimated":""}`);track.setAttribute("role","progressbar");track.setAttribute("aria-valuemin","0");track.setAttribute("aria-valuemax","100");track.setAttribute("aria-valuenow",String(Math.round(progress.percent)));const fill=node("div","progress-fill");fill.style.width=`${progress.percent}%`;track.append(fill);wrap.append(labels,track);row.append(wrap);}
  return row;
}
async function renderActive(data) {
  const section=$("active-section"); if(!data.active_job){section.hidden=true; $("active-log").textContent=""; return;}
  section.hidden=false; const job=data.active_job; const box=$("active-job");let log="";
  try { log=await api(`/admin/api/jobs/${job.id}/log`); $("active-log").textContent=log; $("active-log").scrollTop=$("active-log").scrollHeight; } catch(error){ log=""; $("active-log").textContent=error.message; }
  box.replaceChildren(jobElement(job,progressEstimate(job,log)));
  const stop=node("button","danger","停止する"); stop.addEventListener("click",()=>stopJob(job)); box.append(stop);
}
function legacyMissingFileMessages(log) {
  const keys=[...new Set([...String(log).matchAll(/Eligible attachment ([A-Za-z0-9]+) has no readable local file/g)].map(match=>match[1]))];
  if(!keys.length)return [];
  const messages=keys.map(key=>{
    const location=String(log).match(new RegExp("file://[^\\n']*/storage/"+key+"/([^\\n']+)"));
    let title=key;
    if(location) { try { title=decodeURIComponent(location[1]); } catch { title=key; } }
    return `[停止] 『${title}』（添付 ${key}）の原本ファイルを取得できないため、更新を停止しました。`;
  });
  messages.push("[必要な作業] Zoteroで対象資料の添付を開き、ファイルのダウンロード・同期と保存場所を確認してください。見つからない場合は元ファイルを再配置・再添付し、Zoteroで開けるようになってからクイック実行を再実行してください。");
  return messages;
}
function failureMessages(log) {
  const lines=String(log).split("\n");
  const messages=lines.filter(line=>/^\[(監査失敗|必要な作業|停止|同期待ち)\]/.test(line));
  if(messages.length)return messages;
  const missing=legacyMissingFileMessages(log);
  if(missing.length)return missing;
  for(const line of lines.slice().reverse()) {
    if(!line.startsWith("{"))continue;
    let report; try { report=JSON.parse(line); } catch { continue; }
    if(report.passed!==false)continue;
    if(report.lost_pages_total) messages.push(`[監査失敗] ${report.documents_with_lost_pages}件の資料で、本文が索引にないページが${report.lost_pages_total}頁あります。`);
    if(report.global_failures?.length) messages.push(`[監査失敗] 索引の整合性検査で不合格：${report.global_failures.join("、")}`);
    for(const [key,label] of [["orphan_chunks","原本と対応しない本文"],["dangling_node_chunks","文書構造との不整合"],["unretrievable_items","検索不能な資料"],["unreadable_source_documents","読み取れない原本"]]) {
      if(report[key])messages.push(`[監査失敗] ${label}：${report[key]}件`);
    }
    if(messages.length)messages.push("[必要な作業] 対象資料の取込状態を確認してください。OCR待ちならMistral OCRの回収・採用を完了後、DB監査を再実行してください。有料APIの確認が必要です。");
    break;
  }
  return messages;
}
function nativeCrashMessages(job) {
  if(!(job.exit_code < 0))return [];
  const signal={10:"SIGBUS",11:"SIGSEGV",9:"SIGKILL"}[-job.exit_code] || `シグナル ${-job.exit_code}`;
  return [
    `[停止] 「${job.current_step || job.label}」の処理が${signal}で異常終了しました。`,
    "[必要な作業] 書込み途中の可能性があります。DB監査で状態を確認し、修復・再実行の前にDB一式をバックアップしてください。原本と途中取込の記録は保持してください。"
  ];
}
async function showFailure(job, target) {
  try {
    const log=await api(`/admin/api/jobs/${job.id}/log`);
    const messages=[...nativeCrashMessages(job),...failureMessages(log)];
    target.replaceChildren();
    messages.forEach(message=>target.append(node("p","freshness-warning",message)));
    if(!messages.length) target.append(node("p","freshness-warning",`「${job.current_step || job.label}」で停止しました（終了コード ${text(job.exit_code)}）。下の実行ログを確認してください。`));
    const details=node("details"), summary=node("summary","","実行ログを見る"), output=node("pre","",log);
    details.append(summary,output); target.append(details);
  } catch(error) { target.textContent=`失敗理由を取得できません：${error.message}`; }
}
function renderHistory(data) {
  const signature=JSON.stringify(data.jobs);
  if(signature===historySignature)return;
  historySignature=signature;
  const target=$("history"); target.replaceChildren();
  if(!data.jobs.length) target.append(node("p","","履歴はありません。"));
  data.jobs.forEach(job=>{
    const row=jobElement(job); target.append(row);
    if(job.status==="failed") {
      const explanation=node("div","job-failure","失敗理由を確認中…"); target.append(explanation);
      void showFailure(job,explanation);
    }
  });
}
async function startDefinition(def, paidConsent=false) {
  try { await api("/admin/api/jobs",{method:"POST",body:JSON.stringify({job_type:def.key,selected_steps:def.selected_steps,paid_consent:paidConsent})}); $("notice").textContent=`${def.label}を開始しました。`; await refresh(); }
  catch(error){ $("notice").textContent=error.message; }
}
function openConfirmation(def) {
  if(!def.paid) { void startDefinition(def); return; }
  selectedDefinition=def; $("confirm-title").textContent="API課金の確認";
  $("confirm-description").textContent=`${def.description}。API課金が行われる可能性があります。本当に実行してよいですか？`;
  $("confirm-consent").checked=false; $("confirm-button").disabled=true; $("confirm-dialog").returnValue=""; $("confirm-dialog").showModal();
}
$("confirm-consent").addEventListener("change",()=>{$("confirm-button").disabled=!$("confirm-consent").checked;});
$("confirm-form").addEventListener("submit",event=>{ event.preventDefault(); if($("confirm-consent").checked)$("confirm-dialog").close("confirm"); });
$("confirm-cancel").addEventListener("click",()=>$("confirm-dialog").close("cancel"));
$("confirm-dialog").addEventListener("close",async()=>{
  const definition=selectedDefinition; selectedDefinition=null;
  if($("confirm-dialog").returnValue==="confirm"&&definition&&$("confirm-consent").checked)await startDefinition(definition,true);
});
async function stopJob(job) { if(!confirm("実行中の処理を停止しますか？"))return; try{await api(`/admin/api/jobs/${job.id}/stop`,{method:"POST",body:JSON.stringify({confirmation:"STOP"})}); $("notice").textContent="停止を要求しました。"; await refresh();}catch(error){$("notice").textContent=error.message;} }
async function refresh() { try { state=await api("/admin/api/status"); renderFreshness(state); renderMetrics(state); renderActions(state); await renderActive(state); renderHistory(state); } catch(error){ $("notice").textContent=error.message; } }
refresh(); setInterval(refresh,2500);
