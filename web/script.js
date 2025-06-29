// 确保DOM加载完毕后再执行脚本
document.addEventListener('DOMContentLoaded', () => {

  const $ = id => document.getElementById(id);

  // --- 模式切换相关元素和逻辑 (保持不变) ---
  const modeSwitcher = $("modeSwitcher");
  const videoPanel = $("video-search-panel");
  const imagePanel = $("image-search-panel");

  modeSwitcher.addEventListener('change', (e) => {
    const selectedMode = e.target.value;
    if (selectedMode === 'video') {
      videoPanel.classList.remove('hidden');
      imagePanel.classList.add('hidden');
    } else {
      videoPanel.classList.add('hidden');
      imagePanel.classList.remove('hidden');
    }
  });


  // --- 视频相关元素 ---
  const fInput = $("file"), fName = $("fileName"), query = $("text");
  const btn = $("upload"), wrap = $("progressWrap"), bar = $("progBar"), pText = $("progText");
  const player = $("player"), dl = $("download");
  const videoThresholdInput = $("videoThreshold"); // <-- 新增：获取视频阈值输入框

  // --- 图片相关元素 ---
  const imgInput = $("imageFile"), imgName = $("imageFileName"), imgQuery = $("imageQuery");
  const imgBtn = $("imageSearchBtn"), imgStatus = $("imageStatus");
  const imgResultWrap = $("imageResultWrap"), resImg = $("resultImage"), imgDl = $("downloadImage");
  const thresholdInput = $("threshold");

  // --- 视频相关逻辑 (修改) ---
  fInput.onchange = () => { fName.textContent = fInput.files[0]?.name || "未选择"; };
  btn.onclick = () => {
    if (!fInput.files.length) return alert("请选择视频！");
    if (!query.value.trim()) return alert("请输入检索文字！");

    const file = fInput.files[0];
    const videoThreshold = videoThresholdInput.value; // <-- 新增：获取视频阈值

    wrap.hidden = false; bar.value = 0; pText.textContent = "上传中…";
    player.hidden = true; dl.hidden = true;

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/search", true);
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) {
        const pct = Math.round(e.loaded * 100 / e.total);
        bar.value = pct; pText.textContent = `上传中… ${pct}%`;
      }
    };
    xhr.responseType = "json";
    xhr.onload = () => {
      if (xhr.status !== 200) {
          xhr.response ? alert(`后端错误: ${xhr.response.detail || xhr.status}`) : alert(`后端错误: ${xhr.status}`);
          wrap.hidden = true;
          return;
      }
      const tid = xhr.response.task_id;
      listenProgress(tid);
    };
    xhr.onerror = () => { alert("网络错误"); wrap.hidden = true; };

    const form = new FormData();
    form.append("video", file);
    form.append("query", query.value.trim());
    form.append("threshold", videoThreshold); // <-- 新增：将阈值添加到表单数据中
    xhr.send(form);
  };

  function listenProgress(tid) {
    pText.textContent = "推理中…";
    const es = new EventSource(`/progress/${tid}`);
    es.onmessage = ev => {
      const data = ev.data;
      if (!isNaN(data)) { bar.value = data; }
      if (data.startsWith("('DONE'") || data.startsWith("('ERR'")) {
        es.close(); fetchResult(tid);
      }
    };
    es.onerror = () => {
        es.close();
        pText.textContent = "连接中断，请重试";
    }
  }

  function fetchResult(tid) {
    fetch(`/result/${tid}`).then(r => {
      if (r.status === 202) { setTimeout(() => fetchResult(tid), 1000); return; }
      if (!r.ok) { r.json().then(err => alert(`获取结果失败: ${err.detail}`)); return; }
      return r.blob();
    }).then(blob => {
      if (!blob) return;
      wrap.hidden = true;
      const url = URL.createObjectURL(blob);
      player.src = url; player.hidden = false;
      player.onerror = () => {
        player.hidden = true; dl.hidden = false;
        dl.href = url; dl.download = "result.mp4";
      };
    });
  }

  // --- 图片相关逻辑 (保持不变) ---
  imgInput.onchange = () => { imgName.textContent = imgInput.files[0]?.name || "未选择"; };

  imgBtn.onclick = async () => {
    if (!imgInput.files.length) return alert("请选择图片！");
    if (!imgQuery.value.trim()) return alert("请输入检索文字！");

    const file = imgInput.files[0];
    const query = imgQuery.value.trim();
    const threshold = thresholdInput.value;

    imgBtn.disabled = true;
    imgBtn.textContent = "正在检索...";
    imgStatus.textContent = "正在上传并推理，请稍候...";
    imgResultWrap.hidden = true;

    const formData = new FormData();
    formData.append("image", file);
    formData.append("query", query);
    formData.append("threshold", threshold);

    try {
      const response = await fetch("/search_image", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`检索失败: ${response.status}\n${errorData.detail || JSON.stringify(errorData)}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      resImg.src = url;
      imgDl.href = url;
      imgDl.download = `result_${file.name}`;
      imgResultWrap.hidden = false;
      imgStatus.textContent = "检索完成！";
    } catch (error) {
      console.error("Image search error:", error);
      imgStatus.textContent = `错误: ${error.message}`;
      alert(`发生错误:\n\n${error.message}`);
    } finally {
      imgBtn.disabled = false;
      imgBtn.textContent = "上传并检索 (图片)";
    }
  };
});