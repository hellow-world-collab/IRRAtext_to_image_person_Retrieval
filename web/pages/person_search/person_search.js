// 文件名: web/pages/person_search/person_search.js (已修正)
// 描述: 修正了ID选择器和移除了末尾多余的括号

function initPersonSearch() {
    // 1. 修正选择器：直接在当前加载的HTML片段 (page-container) 中查找元素
    const personSearchPage = document.querySelector('#page-container .page-content.card');
    
    if (personSearchPage) {
        const modeSwitcher = personSearchPage.querySelector("#modeSwitcher");
        const videoPanel = personSearchPage.querySelector("#video-search-panel");
        const imagePanel = personSearchPage.querySelector("#image-search-panel");

        if(modeSwitcher) {
            modeSwitcher.addEventListener('change', (e) => {
                videoPanel.classList.toggle('hidden', e.target.value !== 'video');
                imagePanel.classList.toggle('hidden', e.target.value !== 'image');
            });
        }

        // 视频检索元素和逻辑
        const fInput = personSearchPage.querySelector("#file"), fName = personSearchPage.querySelector("#fileName"), query = personSearchPage.querySelector("#text");
        const btn = personSearchPage.querySelector("#upload"), wrap = personSearchPage.querySelector("#progressWrap"), bar = personSearchPage.querySelector("#progBar"), pText = personSearchPage.querySelector("#progText");
        const player = personSearchPage.querySelector("#player"), dl = personSearchPage.querySelector("#download");
        const videoThresholdInput = personSearchPage.querySelector("#videoThreshold");

        if(fInput) fInput.onchange = () => { fName.textContent = fInput.files[0]?.name || "未选择"; };
        if(btn) btn.onclick = () => {
            if (!fInput.files.length || !query.value.trim()) return alert("请选择视频并输入检索文字！");
            const form = new FormData();
            form.append("video", fInput.files[0]);
            form.append("query", query.value.trim());
            form.append("threshold", videoThresholdInput.value);

            wrap.hidden = false; bar.value = 0; pText.textContent = "上传中…";
            player.hidden = true; dl.hidden = true; btn.disabled = true;

            const xhr = new XMLHttpRequest();
            xhr.open("POST", "/search", true);
            xhr.upload.onprogress = e => { if (e.lengthComputable) bar.value = Math.round(e.loaded * 100 / e.total); };
            xhr.responseType = "json";
            xhr.onload = () => {
                btn.disabled = false;
                if (xhr.status !== 200) {
                    alert(`后端错误: ${xhr.response?.detail || xhr.status}`);
                    wrap.hidden = true; return;
                }
                listenPersonSearchProgress(xhr.response.task_id);
            };
            xhr.onerror = () => { alert("网络错误"); wrap.hidden = true; btn.disabled = false;};
            xhr.send(form);
        };

        function listenPersonSearchProgress(tid) {
            pText.textContent = "服务器推理中…";
            const es = new EventSource(`/progress/${tid}`);
            es.onmessage = ev => bar.value = parseInt(ev.data, 10);
            es.addEventListener('done', (ev) => {
                es.close();
                const ossUrl = ev.data;
                wrap.hidden = true;
                player.src = ossUrl;
                player.hidden = false;
                dl.href = ossUrl;
                dl.download = ossUrl.split('/').pop();
                dl.hidden = false;
            });
            es.addEventListener('err', (ev) => {
                es.close();
                pText.textContent = `处理失败: ${ev.data}`;
            });
            es.onerror = () => { es.close(); };
        }

        const imgInput = personSearchPage.querySelector("#imageFile"), imgName = personSearchPage.querySelector("#imageFileName"), imgQuery = personSearchPage.querySelector("#imageQuery");
        const imgBtn = personSearchPage.querySelector("#imageSearchBtn"), imgStatus = personSearchPage.querySelector("#imageStatus");
        const imgResultWrap = personSearchPage.querySelector("#imageResultWrap"), resImg = personSearchPage.querySelector("#resultImage"), imgDl = personSearchPage.querySelector("#downloadImage");
        const thresholdInput = personSearchPage.querySelector("#threshold");

        if(imgInput) imgInput.onchange = () => { imgName.textContent = imgInput.files[0]?.name || "未选择"; };
        if(imgBtn) imgBtn.onclick = async () => {
            if (!imgInput.files.length || !imgQuery.value.trim()) return alert("请选择图片并输入检索文字！");
            const formData = new FormData();
            formData.append("image", imgInput.files[0]);
            formData.append("query", imgQuery.value.trim());
            formData.append("threshold", thresholdInput.value);
            imgBtn.disabled = true; imgStatus.textContent = "正在检索..."; imgResultWrap.hidden = true;
            try {
                const response = await fetch("/search_image", { method: "POST", body: formData });
                if (!response.ok) throw new Error((await response.json()).detail);
                const data = await response.json();
                const ossUrl = data.result_url;
                resImg.src = ossUrl;
                imgDl.href = ossUrl;
                imgDl.download = `result_${imgInput.files[0].name}`;
                imgResultWrap.hidden = false;
                imgStatus.textContent = "检索完成！";
            } catch (error) {
                imgStatus.textContent = `错误: ${error.message}`;
            } finally {
                imgBtn.disabled = false;
            }
        };
    }
}

// 2. 移除多余的括号，并正确调用函数
initPersonSearch();