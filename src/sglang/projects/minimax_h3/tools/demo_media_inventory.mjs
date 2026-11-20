import {readFile,writeFile,stat,readdir} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),base=path.join(root,'web/demos');
const data=await import(pathToFileURL(path.join(base,'manifest.mjs')).href);
const boxes=new Map();const document={getElementById(id){if(!boxes.has(id))boxes.set(id,{innerHTML:''});return boxes.get(id)},querySelectorAll(){return[]}};
const src=(await readFile(path.join(base,'app.mjs'),'utf8')).replace(/^import .*;\r?\n/,'');
vm.runInNewContext(src,{...data,document});
const refs=[];
for(const [section,box] of boxes){for(const tag of box.innerHTML.matchAll(/<(img|video)\b[^>]*>/g)){for(const field of ['src','poster']){const m=tag[0].match(new RegExp(`\\b${field}="([^"]+)"`));if(!m)continue;const p=path.resolve(base,m[1].split('?')[0]);refs.push({section,tag:tag[1],field,path:path.relative(root,p).replaceAll('\\','/')});}}}
const unique=new Map();for(const r of refs){if(!unique.has(r.path))unique.set(r.path,{path:r.path,uses:[]});unique.get(r.path).uses.push({section:r.section,tag:r.tag,field:r.field});}
const files=[];for(const r of unique.values()){try{const s=await stat(path.join(root,r.path));files.push({...r,bytes:s.size,sha256:createHash('sha256').update(await readFile(path.join(root,r.path))).digest('hex')})}catch{files.push({...r,missing:true,bytes:0})}}
const sections={};for(const name of boxes.keys()){const list=files.filter(f=>f.uses.some(u=>u.section===name));sections[name]={files:list.length,bytes:list.reduce((n,f)=>n+f.bytes,0),videos:list.filter(f=>f.path.endsWith('.mp4')).length,missing:list.filter(f=>f.missing).length}}
const result={files,sections,uniqueFiles:files.length,totalBytes:files.reduce((n,f)=>n+f.bytes,0),videoBytes:files.filter(f=>f.path.endsWith('.mp4')).reduce((n,f)=>n+f.bytes,0),videos:files.filter(f=>f.path.endsWith('.mp4')).length,missing:files.filter(f=>f.missing)};
if(result.missing.length || files.some(f=>f.path.startsWith('.local/') || f.path.startsWith('../')))throw new Error('Media must exist inside the case asset tree; restore external videos before rebuilding this full inventory');
await writeFile(path.join(base,'media-inventory.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({...result,files:undefined},null,2));


// Keep video bytes outside Git while retaining a complete integrity manifest.
async function* videoFiles(folder) {
  for (const item of await readdir(folder,{withFileTypes:true})) {
    const file=path.join(folder,item.name);
    if(item.isDirectory()) {
      if(item.name!=='frames' && !item.name.endsWith('_frames') && item.name!=='__pycache__' && item.name!=='.audit') yield* videoFiles(file);
    } else if(/\.(mp4|mkv|wav|webm|mov|avi|gif)$/i.test(item.name)) yield file;
  }
}
const external=[];
for await (const file of videoFiles(path.join(root,'docs/demo-brainstorm'))) {
  const data=await readFile(file);
  external.push({path:path.relative(root,file).replaceAll('\\','/'),bytes:data.length,sha256:createHash('sha256').update(data).digest('hex')});
}
external.sort((a,b)=>a.path.localeCompare(b.path,'en'));
const manifest={schema:1,storage:{status:'local-only',archive_uri:null,restore:'Restore files at their exact repository-relative paths and verify SHA-256. External archive has not yet been registered.'},fileCount:external.length,totalBytes:external.reduce((n,f)=>n+f.bytes,0),files:external};
await writeFile(path.join(root,'docs/demo-brainstorm/media-artifacts.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({externalFiles:manifest.fileCount,externalBytes:manifest.totalBytes}));
