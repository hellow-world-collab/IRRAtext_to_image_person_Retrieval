const $ = id => document.getElementById(id);

const fInput=$("file"), fName=$("fileName"), query=$("text");
const btn=$("upload"), wrap=$("progressWrap"), bar=$("progBar"), pText=$("progText");
const player=$("player"), dl=$("download");

fInput.onchange = ()=>{ fName.textContent = fInput.files[0]?.name || "未选择"; };

btn.onclick = ()=>{
  if(!fInput.files.length) return alert("请选择视频！");
  if(!query.value.trim())  return alert("请输入检索文字！");
  const file=fInput.files[0];

  // 重置 UI
  wrap.hidden=false; bar.value=0; pText.textContent="上传中…";
  player.hidden=true; dl.hidden=true;

  // ---------- 执行上传 ----------
  const xhr=new XMLHttpRequest();
  xhr.open("POST","/search",true);
  xhr.upload.onprogress=e=>{
    if(e.lengthComputable){
      const pct=Math.round(e.loaded*100/e.total);
      bar.value=pct;
      pText.textContent=`上传中… ${pct}%`;
    }
  };
  xhr.responseType="json";
  xhr.onload=()=>{
    if(xhr.status!==200){ alert("后端错误: "+xhr.status); return; }
    const tid=xhr.response.task_id;
    listenProgress(tid);
  };
  xhr.onerror = ()=> alert("网络错误");
  const form=new FormData();
  form.append("video",file); form.append("query",query.value.trim());
  xhr.send(form);
};

function listenProgress(tid){
  pText.textContent="推理中…";
  const es=new EventSource(`/progress/${tid}`);
  es.onmessage = ev=>{
    const data=ev.data;
    if(!isNaN(data)){ bar.value=data; }
    if(data.startsWith("('DONE'") || data.startsWith("('ERR'")){
      es.close(); fetchResult(tid);
    }
  };
}
function fetchResult(tid){
  fetch(`/result/${tid}`).then(r=>{
    if(r.status===202){ setTimeout(()=>fetchResult(tid),1000); return;}
    if(r.status!==200){ r.text().then(t=>alert(t)); return;}
    return r.blob();
  }).then(blob=>{
    if(!blob) return;
    wrap.hidden=true;
    const url=URL.createObjectURL(blob);
    player.src=url; player.hidden=false;
    player.onerror=()=>{ player.hidden=true; dl.hidden=false;
      dl.href=url; dl.download="result.mp4"; };
  });
}
