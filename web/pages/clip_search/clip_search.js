// 文件名: web/pages/clip_search/clip_search.js (已修正)
// 描述: 修正了ID选择器和移除了末尾多余的括号，修复了上传功能

function initClipSearch() {
    // 1. 修正选择器：直接在当前加载的HTML片段中查找元素
    const clipPage = document.querySelector('#page-container .page-content.card');

    if (clipPage) {
        const clipFileInput = clipPage.querySelector("#clipFile"), clipFileName = clipPage.querySelector("#clipFileName");
        const clipQueryInput = clipPage.querySelector("#clipQuery"), clipTopNInput = clipPage.querySelector("#clipTopN");
        const clipThresholdInput = clipPage.querySelector("#clipThreshold");
        const clipUploadBtn = clipPage.querySelector("#clipUploadBtn");
        const clipProgressWrap = clipPage.querySelector("#clipProgressWrap"), clipProgBar = clipPage.querySelector("#clipProgBar"), clipProgText = clipPage.querySelector("#clipProgText");
        const clipResultsWrap = clipPage.querySelector("#clipResultsWrap");

        if(clipFileInput) clipFileInput.onchange = () => { clipFileName.textContent = clipFileInput.files[0]?.name || "未选择"; };
        if(clipUploadBtn) clipUploadBtn.onclick = () => {
            if (!clipFileInput.files.length || !clipQueryInput.value.trim()) return alert("请选择视频并输入检索文字！");
            const formData = new FormData();
            formData.append("video", clipFileInput.files[0]);
            formData.append("query", clipQueryInput.value.trim());
            formData.append("top_n", clipTopNInput.value);
            formData.append("threshold", clipThresholdInput.value);
            clipProgressWrap.hidden = false; clipProgBar.value = 0; clipProgText.textContent = "上传中...";
            clipUploadBtn.disabled = true; clipResultsWrap.innerHTML = '';
            const xhr = new XMLHttpRequest();
            xhr.open("POST", "/clip/search", true);
            xhr.upload.onprogress = e => {
                if (e.lengthComputable) {
                    const pct = Math.round((e.loaded / e.total) * 100);
                    clipProgBar.value = pct;
                    if (pct === 100) clipProgText.textContent = "服务器处理中，请稍候...";
                }
            };
            xhr.responseType = "json";
            xhr.onload = () => {
                clipUploadBtn.disabled = false;
                clipProgressWrap.hidden = true;
                if (xhr.status!== 200) return alert(`错误: ${xhr.response?.detail || xhr.statusText}`);
                displayClipResults(xhr.response.results);
            };
            xhr.onerror = () => {
                alert("网络错误");
                clipUploadBtn.disabled = false;
                clipProgressWrap.hidden = true;
            };
            xhr.send(formData);
        };

        function displayClipResults(results) {
            if (!results || results.length === 0) {
                clipResultsWrap.innerHTML = '<p class="info">未能找到高于设定阈值的匹配片段。</p>';
                return;
            }
            results.forEach(res => {
                const item = document.createElement('div');
                item.className = 'clip-result-item';
                item.innerHTML = `
                    <h4>排名 ${res.rank} (相似度: ${res.similarity.toFixed(4)})</h4>
                    <p>时间段: ${res.start_time.toFixed(2)}s - ${res.end_time.toFixed(2)}s</p>
                    <video src="${res.video_url}" controls preload="metadata"></video>
                    <a href="${res.video_url}" download>下载此片段</a>
                `;
                clipResultsWrap.appendChild(item);
            });
        }
    }
}

// 2. 移除多余的括号，并正确调用函数
initClipSearch();