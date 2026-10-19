from __future__ import annotations
from unittest.mock import AsyncMock,Mock
import fitz
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from citation_graph.debug_routes import register_debug_routes
from citation_graph.debug_reader import DebugReader


@pytest.fixture
def debug_client(tmp_path,monkeypatch):
    path=tmp_path/'source.pdf'
    with fitz.open() as doc:
        doc.new_page().insert_text((40,40),'Original');doc.save(path)
    rows=[{'id':'C1','text':'Original','metadata':{'page':1,'attachmentKey':'ABCD1234','title':'Book'}}]
    import citation_graph.debug_reader as module
    monkeypatch.setattr(module,'get_item_chunks',lambda *a,**kw:rows)
    monkeypatch.setattr(module,'load_manifest',lambda p:{'files':{}})
    monkeypatch.setattr(DebugReader,'resolve',AsyncMock(return_value=(path,'ITEM1234')))
    monkeypatch.setattr(DebugReader,'inventory',lambda self:[{'attachment_key':'ABCD1234','title':'Book','issues':['OCR']}])
    static=tmp_path/'citation_graph/static';static.mkdir(parents=True);(static/'debug.html').write_text('<h1>Review</h1>')
    app=FastAPI()
    register_debug_routes(app,tmp_path,
        lambda req:None if req.headers.get('authorization')=='test' else JSONResponse({'error':'auth'},status_code=401),
        lambda req:None if req.headers.get('origin')=='https://review.test' else JSONResponse({'error':'origin'},status_code=403),
        lambda req:'reviewer')
    return TestClient(app),rows


def test_review_routes_auth_origin_versions_images_and_export(debug_client,monkeypatch):
    client,rows=debug_client
    base='/admin/debug/api/documents/ABCD1234'
    headers={'Authorization':'test','Origin':'https://review.test'}
    for url in ['/admin/debug/', '/admin/debug/api/documents',base+'/pages/1',base+'/pages/1/image?version=x',base+'/draft/export',base+'/diff?revision=1']:
        assert client.get(url).status_code==401
    assert client.post(base+'/draft',json={'revision':0,'version':'x'*64}).status_code==401
    assert client.post(base+'/apply',json={'revision':1,'version':'x'*64}).status_code==401
    assert client.get('/admin/debug/',headers=headers).status_code==200
    assert client.get('/admin/debug/api/documents',headers=headers).json()[0]['title']=='Book'
    data=client.get(base+'/pages/1',headers=headers).json();assert 'all_chunks' not in data
    version=data['version']
    image=client.get(base+f'/pages/1/image?version={version}',headers=headers)
    assert image.status_code==200 and image.headers['content-type']=='image/jpeg'
    assert client.get(base+'/pages/1/image?version=old',headers=headers).status_code==409
    assert client.get(base+'/pages/2',headers=headers).status_code==400
    body={'revision':0,'version':version,'edits':{'C1':'Corrected'},'toc':[]}
    assert client.post(base+'/draft',json=body,headers={'Authorization':'test'}).status_code==403
    saved=client.post(base+'/draft',json=body,headers=headers);assert saved.status_code==200
    assert client.post(base+'/draft',json=body,headers=headers).status_code==409
    assert client.get(base+'/draft/export',headers=headers).json()['revision']==1
    diff=client.get(base+'/diff?revision=1',headers=headers).json();assert diff['text_changes'][0]['before']=='Original'
    assert client.get(base+'/diff?revision=2',headers=headers).status_code==409
    import citation_graph.admin_jobs as jobs
    start=Mock(return_value={'id':'JOB','status':'queued','token':'SECRET'})
    monkeypatch.setattr(jobs,'start_job',start)
    body['revision']=1
    applied=client.post(base+'/apply',json=body,headers=headers)
    assert applied.status_code==202 and 'token' not in applied.json()
    assert start.call_args.kwargs['debug_selection']=={'attachment_key':'ABCD1234','revision':1}
    body['version']='x'*64
    assert client.post(base+'/apply',json=body,headers=headers).status_code==409
    rows[0]['text']='Changed'
    assert client.post(base+'/draft',json={**body,'version':version},headers=headers).status_code==409


def test_history_restore_appends_versions_without_erasing_history(debug_client):
    client,rows=debug_client
    base='/admin/debug/api/documents/ABCD1234'
    headers={'Authorization':'test','Origin':'https://review.test'}
    version=client.get(base+'/pages/1',headers=headers).json()['version']
    body={'revision':0,'version':version,'edits':{'C1':'Corrected'},'toc':[]}
    for suffix in ['/history','/history/1']:
        assert client.get(base+suffix).status_code==401
    assert client.get(base+'/history',headers=headers).json()==[]
    assert client.get(base+'/history/1',headers=headers).status_code==404
    first=client.post(base+'/draft',json=body,headers=headers).json()
    history=client.get(base+'/history',headers=headers).json()
    assert history[0]['actor']=='reviewer' and history[0]['created_at']
    revision=client.get(base+'/history/1',headers=headers).json()
    assert revision['originals']=={'C1':'Original'}
    assert revision['edits']=={'C1':'Corrected'}
    undo={**body,'revision':first['revision']}
    assert client.post(base+'/history/1/restore',json=undo).status_code==401
    assert client.post(base+'/history/1/restore',json=undo,headers={'Authorization':'test'}).status_code==403
    second=client.post(base+'/history/1/restore?before=true',json=undo,headers=headers).json()
    assert second['revision']==2 and second['edits']=={'C1':'Original'}
    third=client.post(base+'/history/1/restore?before=false',json={**body,'revision':2},headers=headers).json()
    assert third['revision']==3 and third['edits']=={'C1':'Corrected'}
    assert client.get(base+'/history?before=3',headers=headers).json()[0]['revision']==2
    assert client.get(base+'/history/1',headers=headers).json()==revision
    assert rows[0]['text']=='Original'  # restoring a draft never mutates the index
    assert client.post(base+'/history/1/restore',json=undo,headers=headers).status_code==409
    assert client.post(base+'/history/99/restore',json=undo,headers=headers).status_code==404


def test_source_changes_and_missing_sources_fail_visibly(debug_client,monkeypatch):
    client,_=debug_client
    base='/admin/debug/api/documents/ABCD1234'
    headers={'Authorization':'test','Origin':'https://review.test'}
    data=client.get(base+'/pages/1',headers=headers).json()
    version=data['version']
    body={'revision':0,'version':version,'edits':{'C1':'Corrected'},'toc':[]}
    client.post(base+'/draft',json=body,headers=headers)
    original=DebugReader.inspect
    count=0
    def inspect(self,*args):
        nonlocal count
        count+=1
        result=original(self,*args)
        if count>1:result['version']='changed'
        return result
    monkeypatch.setattr(DebugReader,'inspect',inspect)
    assert client.get(base+f'/pages/1/image?version={version}',headers=headers).status_code==409
    def different_source(self,*args):
        result=original(self,*args);result['source_sha256']='changed';return result
    monkeypatch.setattr(DebugReader,'inspect',different_source)
    assert client.post(base+'/history/1/restore',json={**body,'revision':1},headers=headers).status_code==409
    monkeypatch.setattr(DebugReader,'resolve',AsyncMock(side_effect=FileNotFoundError('同期してください')))
    response=client.get(base+'/draft/export',headers=headers)
    assert response.status_code==404 and '同期' in response.json()['error']


def test_debug_worker_persists_exact_revision_and_runs_apply_then_audit(tmp_path,monkeypatch):
    import os
    from types import SimpleNamespace
    from citation_graph import admin_jobs as jobs
    python=tmp_path/'.venv/bin/python';python.parent.mkdir(parents=True);python.touch()
    monkeypatch.setattr(jobs,'active_job',lambda root:None)
    commands=[]
    def spawn(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(pid=os.getpid(),wait=lambda:0)
    monkeypatch.setattr(jobs.subprocess,'Popen',spawn)
    selection={'attachment_key':'ABCD1234','revision':3}
    record=jobs.start_job('debug_apply','','reviewer',tmp_path,debug_selection=selection)
    assert record['debug_selection']==selection and not record['paid_consent']
    assert jobs.run_job(record['id'],'debug_apply',record['token'],tmp_path)==0
    assert commands[1][-4:]==('--attachment','ABCD1234','--revision','3')
    assert commands[2][1].endswith('run_db_audit.py')
    assert jobs.read_record(record['id'],tmp_path)['status']=='completed'


def test_zone_history_restore_and_unresolved_toc_blocks_apply(debug_client,monkeypatch):
    client,rows=debug_client
    base='/admin/debug/api/documents/ABCD1234'
    headers={'Authorization':'test','Origin':'https://review.test'}
    version=client.get(base+'/pages/1',headers=headers).json()['version']
    body={'revision':0,'version':version,'zones':{'C1':'bibliography'}}
    response=client.post(base+'/draft',json=body,headers=headers)
    assert response.status_code==200
    diff=client.get(base+'/diff?revision=1',headers=headers).json()
    assert diff['zone_changes']==[{'chunk_id':'C1','before':'body','after':'bibliography'}]
    undo=client.post(base+'/history/1/restore',json={**body,'revision':1},headers=headers).json()
    assert undo['zones']=={'C1':'body'} and undo['revision']==2
    redo=client.post(base+'/history/1/restore?before=false',json={**body,'revision':2},headers=headers).json()
    assert redo['zones']=={'C1':'bibliography'}
    assert client.get(base+'/history',headers=headers).json()[0]['zone_changes']==1
    rows[0]['metadata']['page']=None
    data=client.get(base+'/pages/1',headers=headers).json()
    body={'revision':3,'version':data['version'],'toc':[{'page':1,'level':1,'title':'Chapter'}],'toc_changed':True}
    saved=client.post(base+'/draft',json=body,headers=headers)
    assert saved.status_code==200 and 'chunk_id' not in saved.json()['toc'][0]
    from citation_graph import admin_jobs
    start=Mock();monkeypatch.setattr(admin_jobs,'start_job',start)
    applied=client.post(base+'/apply',json={**body,'revision':4},headers=headers)
    assert applied.status_code==400 and 'チャンク' in applied.json()['error']
    start.assert_not_called()


def test_empty_saved_draft_never_starts_a_worker(debug_client,monkeypatch):
    client,_=debug_client
    base='/admin/debug/api/documents/ABCD1234'
    headers={'Authorization':'test','Origin':'https://review.test'}
    version=client.get(base+'/pages/1',headers=headers).json()['version']
    client.post(base+'/draft',json={'revision':0,'version':version},headers=headers)
    from citation_graph import admin_jobs
    start=Mock();monkeypatch.setattr(admin_jobs,'start_job',start)
    response=client.post(base+'/apply',json={'revision':1,'version':version},headers=headers)
    assert response.status_code==400 and '修正がありません' in response.json()['error']
    start.assert_not_called()


def test_split_draft_diff_apply_and_history_rebase_after_db_changes(debug_client,monkeypatch):
    from citation_graph.debug_changes import corrected_rows
    client,rows=debug_client
    rows[0]['text']='Before.\nInaugurations\nText.'
    original=rows[0]['text']
    base='/admin/debug/api/documents/ABCD1234';headers={'Authorization':'test','Origin':'https://review.test'}
    version=client.get(base+'/pages/1',headers=headers).json()['version']
    heading={'chunk_id':'C1','page':1,'level':1,'title':'Inaugurations','offset':8,'anchor_text':original}
    body={'revision':0,'version':version,'toc':[heading],'toc_changed':True}
    saved=client.post(base+'/draft',headers=headers,json=body)
    assert saved.status_code==200,saved.text
    first=saved.json()
    diff=client.get(base+'/diff?revision=1',headers=headers).json()
    assert diff['split_changes'][0]['parts']==['Before.\n','Inaugurations\nText.']
    assert first['originals']=={'C1':original}
    import citation_graph.admin_jobs as jobs
    monkeypatch.setattr(jobs,'start_job',Mock(return_value={'id':'JOB','status':'queued'}))
    assert client.post(base+'/apply',headers=headers,json={**body,'revision':1}).status_code==202
    # Simulate completion of the real split without touching a canonical DB.
    rows[:]=corrected_rows(rows,{},[heading],toc_changed=True)
    current=client.get(base+'/pages/1',headers=headers).json()
    undo=client.post(base+'/history/1/restore?before=true',headers=headers,json={'revision':1,'version':current['version']})
    assert undo.status_code==200,undo.text
    assert undo.json()['merges']==first['split_groups']
    assert undo.json()['edits']=={'C1':original}
    assert client.post(base+'/apply',headers=headers,json={'revision':2,'version':current['version']}).status_code==202
    # Merge, then redo the first split and undo that merge via its own history.
    inverse=undo.json()
    rows[:]=corrected_rows(rows,inverse['edits'],inverse['toc'],merges=inverse['merges'],toc_changed=True)
    version=client.get(base+'/pages/1',headers=headers).json()['version']
    redo=client.post(base+'/history/1/restore?before=false',headers=headers,json={'revision':2,'version':version})
    assert redo.status_code==200,redo.text
    assert redo.json()['toc'][0]['offset']==8
    again=client.post(base+'/history/2/restore?before=true',headers=headers,json={'revision':3,'version':version})
    assert again.status_code==200,again.text
    assert again.json()['cuts']=={'C1':[8]}
    assert again.json()['toc'][0]['offset']==8
    assert len(client.get(base+'/history',headers=headers).json())==4
    # Invalid/stale character positions never make a new saved revision.
    for offset in [True,-1,999,'8']:
        bad=client.post(base+'/draft',headers=headers,json={'revision':4,'version':version,'toc':[{**heading,'offset':offset}]})
        assert bad.status_code==400
    bad=client.post(base+'/draft',headers=headers,json={'revision':4,'version':version,'edits':{'C1':'Changed '+original},'toc':[heading]})
    assert bad.status_code==400 and '再指定' in bad.json()['error']
