"use strict";
const $ = (id) => document.getElementById(id);
let state = null;
let selectedDefinition = null;

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
    metric("DB監査gate",gate.passed?"合格":"未合格",localTime(gate.modified_at)), metric("未解決artifact",a.unresolved ?? "—",a.error || "failed + blocked"),
    metric("索引書き込み",data.indexing_lock?"実行中":"停止中",data.indexing_lock?.started_at || "ロックなし"),
    metric("Mistral Batch",batch?statusLabel(batch.phase):"未実行",batch?`${batch.completed_requests}/${batch.total_requests || batch.candidate_count}件 · 確認 ${localTime(batch.last_checked_at)}`:"")
  ); $("updated").textContent=`更新 ${new Date().toLocaleTimeString("ja-JP")}`;
}
function freshnessCard(label, value, detail, keys=[], attention=0) {
  const count=Number.parseInt(value,10), state=attention>0?"attention":count>0?"pending":"current";
  const box=node("div",`freshness-card ${state}`);
  box.append(node("span","",label),node("b",state,value),node("small","",detail));
  if(keys.length) box.append(node("code","",keys.join(" · "))); return box;
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
    freshnessCard("文書構造・目次",`${text(structure.pending)}件`,structure.attention?`失敗・要確認 ${structure.attention}件`:"保存チャンクと照合",structure.sample_keys||[],structure.attention||0),
    freshnessCard("階層要約",`${text(summaries.pending)}件`,summaries.attention?`要確認 ${summaries.attention}項目・索引不整合 ${summaries.index_issues||0}件`:`要約索引 ${summaries.actual_index_rows||0}件・整合`,summaries.sample_keys||[],summaries.attention||0),
    freshnessCard("Citation Network",`${text(citations.pending)}件`,citations.attention?`うちエラー ${citations.attention}件`:citations.metadata_changed?`書誌情報変更 ${citations.metadata_changed}件`:"未処理・再試行対象",citations.sample_keys||[],citations.attention||0)
  );
  $("freshness-updated").textContent=report.recheck_pending?"再確認待ち":report.stale?"期限切れ":`確認 ${localTime(report.generated_at)}`;
}
function renderActions(data) {
  const target=$("actions"); target.replaceChildren(); const busy=Boolean(data.active_job);
  data.definitions.forEach(def=>{ const card=node("article","action"); card.append(node("h3","",def.label),node("p","",def.description));
    if(def.paid) card.append(node("div","paid","有料APIを使用")); const button=node("button","","実行する"); button.disabled=busy;
    button.addEventListener("click",()=>openConfirmation(def)); card.append(button); target.append(card); });
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
function renderHistory(data) { const target=$("history"); target.replaceChildren(); if(!data.jobs.length) target.append(node("p","","履歴はありません。")); data.jobs.forEach(job=>target.append(jobElement(job))); }
async function startDefinition(def, confirmation="") {
  try { await api("/admin/api/jobs",{method:"POST",body:JSON.stringify({job_type:def.key,confirmation})}); $("notice").textContent=`${def.label}を開始しました。`; await refresh(); }
  catch(error){ $("notice").textContent=error.message; }
}
function openConfirmation(def) {
  if(!def.confirmation) { void startDefinition(def); return; }
  selectedDefinition=def; $("confirm-title").textContent=def.label; $("confirm-description").textContent=def.description;
  $("confirm-label").hidden=false; $("confirm-input").value=""; $("confirm-input").placeholder=def.confirmation; $("confirm-dialog").showModal(); $("confirm-input").focus(); }
$("confirm-form").addEventListener("submit",event=>{ event.preventDefault(); $("confirm-dialog").close("confirm"); });
$("confirm-cancel").addEventListener("click",()=>$("confirm-dialog").close("cancel"));
$("confirm-dialog").addEventListener("close",async()=>{ if($("confirm-dialog").returnValue!=="confirm"||!selectedDefinition){selectedDefinition=null; return;}
  const definition=selectedDefinition; selectedDefinition=null; await startDefinition(definition,$("confirm-input").value);
});
async function stopJob(job) { if(prompt("停止するには STOP と入力してください")!=="STOP")return; try{await api(`/admin/api/jobs/${job.id}/stop`,{method:"POST",body:JSON.stringify({confirmation:"STOP"})}); $("notice").textContent="停止を要求しました。"; await refresh();}catch(error){$("notice").textContent=error.message;} }
async function refresh() { try { state=await api("/admin/api/status"); renderFreshness(state); renderMetrics(state); renderActions(state); await renderActive(state); renderHistory(state); } catch(error){ $("notice").textContent=error.message; } }
refresh(); setInterval(refresh,2500);
