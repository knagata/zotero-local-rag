import base64,hashlib,io,json,tarfile,urllib.request
from pathlib import Path

def main():
    version='6.4.299'
    url=f'https://registry.npmjs.org/pdfjs-dist/-/pdfjs-dist-{version}.tgz'
    expected='AVl138zALtfaAPvADulE0PZThbYzCBS79nL4pOSL/6Sm/4AH5A21BD9VHt97OlCuzJuCpmeZtAtkinisF4Vb1g=='
    raw=urllib.request.urlopen(url,timeout=60).read()
    assert base64.b64encode(hashlib.sha512(raw).digest()).decode()==expected
    root=Path('citation_graph/static/pdfjs');root.mkdir(parents=True,exist_ok=True)
    assets={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
     for member in tar.getmembers():
      name=member.name.removeprefix('package/')
      if name in ('legacy/build/pdf.mjs','legacy/build/pdf.worker.mjs','web/pdf_viewer.css','LICENSE'):
       target=name.rsplit('/',1)[-1]
      elif name.startswith(('cmaps/','standard_fonts/','wasm/','iccs/')) and member.isfile():target=name
      else:continue
      data=tar.extractfile(member).read();path=root/target;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
      assets[target]=hashlib.sha256(data).hexdigest()
    (root/'assets.json').write_text(json.dumps({'version':version,'source':url,'integrity':'sha512-'+expected,'files':assets},indent=2)+'\n')
    print(json.dumps({'version':version,'files':len(assets),'bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}))

if __name__ == "__main__":
    main()
