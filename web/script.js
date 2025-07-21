// 文件名: script.js (完整版)
// 描述: 实现了页面切换、全身行人检索、Clip片段检索和历史记录三大功能

document.addEventListener('DOMContentLoaded', () => {

    const $ = id => document.getElementById(id);

    // --- 全局元素 ---
    const appContainer = $("app-container");
    const menuToggle = $("menu-toggle");
    const menuLinks = document.querySelectorAll('.menu-links li');
    const pages = document.querySelectorAll('.page-content');

    // --- 菜单和页面切换逻辑 ---
    menuToggle.addEventListener('click', () => {
        appContainer.classList.toggle('sidebar-open');
    });

    menuLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageId = link.getAttribute('data-page');
            pages.forEach(page => page.classList.toggle('hidden', page.id !== pageId));
            menuLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            if (pageId === 'history-page') loadHistory();
            if (window.innerWidth < 768) appContainer.classList.remove('sidebar-open');
        });
    });

    // --- 页面1: 全身行人检索 (Person Search) ---
    const personSearchPage = $("person-search-page");
    if (personSearchPage) {
        const modeSwitcher = personSearchPage.querySelector("#modeSwitcher");
        const videoPanel = personSearchPage.querySelector("#video-search-panel");
        const imagePanel = personSearchPage.querySelector("#image-search-panel");

        modeSwitcher.addEventListener('change', (e) => {
            videoPanel.classList.toggle('hidden', e.target.value !== 'video');
            imagePanel.classList.toggle('hidden', e.target.value !== 'image');
        });

        // 视频检索元素和逻辑
        const fInput = personSearchPage.querySelector("#file"), fName = personSearchPage.querySelector("#fileName"), query = personSearchPage.querySelector("#text");
        const btn = personSearchPage.querySelector("#upload"), wrap = personSearchPage.querySelector("#progressWrap"), bar = personSearchPage.querySelector("#progBar"), pText = personSearchPage.querySelector("#progText");
        const player = personSearchPage.querySelector("#player"), dl = personSearchPage.querySelector("#download");
        const videoThresholdInput = personSearchPage.querySelector("#videoThreshold");

        fInput.onchange = () => { fName.textContent = fInput.files[0]?.name || "未选择"; };
        btn.onclick = () => {
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
                fetchPersonSearchResult(ev.data);
            });
            es.addEventListener('err', (ev) => {
                es.close();
                pText.textContent = `处理失败: ${ev.data}`;
            });
            es.onerror = () => { es.close(); };
        }

        function fetchPersonSearchResult(tid) {
            const resultUrl = `/result/${tid}`;
            fetch(resultUrl).then(r => {
                if (!r.ok) { r.json().then(err => alert(`获取结果失败: ${err.detail}`)); return; }
                return r.blob();
            }).then(blob => {
                if (!blob) return;
                wrap.hidden = true;
                const objectURL = URL.createObjectURL(blob);
                player.src = objectURL;
                player.hidden = false;
                dl.href = objectURL;
                dl.download = `result_${tid}.mp4`;
                dl.hidden = false;
            }).catch(err => alert("获取结果文件时发生网络错误。"));
        }

        // 图片检索元素和逻辑
        const imgInput = personSearchPage.querySelector("#imageFile"), imgName = personSearchPage.querySelector("#imageFileName"), imgQuery = personSearchPage.querySelector("#imageQuery");
        const imgBtn = personSearchPage.querySelector("#imageSearchBtn"), imgStatus = personSearchPage.querySelector("#imageStatus");
        const imgResultWrap = personSearchPage.querySelector("#imageResultWrap"), resImg = personSearchPage.querySelector("#resultImage"), imgDl = personSearchPage.querySelector("#downloadImage");
        const thresholdInput = personSearchPage.querySelector("#threshold");

        imgInput.onchange = () => { imgName.textContent = imgInput.files[0]?.name || "未选择"; };
        imgBtn.onclick = async () => {
            if (!imgInput.files.length || !imgQuery.value.trim()) return alert("请选择图片并输入检索文字！");
            const formData = new FormData();
            formData.append("image", imgInput.files[0]);
            formData.append("query", imgQuery.value.trim());
            formData.append("threshold", thresholdInput.value);
            imgBtn.disabled = true; imgStatus.textContent = "正在检索..."; imgResultWrap.hidden = true;
            try {
                const response = await fetch("/search_image", { method: "POST", body: formData });
                if (!response.ok) throw new Error((await response.json()).detail);
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                resImg.src = url;
                imgDl.href = url;
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

    // --- 页面2: 视频片段检索 (已更新) ---
    const clipPage = document.getElementById("clip-search-page");
    if (clipPage) {
        const clipFileInput = clipPage.querySelector("#clipFile"), clipFileName = clipPage.querySelector("#clipFileName");
        const clipQueryInput = clipPage.querySelector("#clipQuery"), clipTopNInput = clipPage.querySelector("#clipTopN");
        const clipThresholdInput = clipPage.querySelector("#clipThreshold"); // 【新增】获取阈值输入框
        const clipUploadBtn = clipPage.querySelector("#clipUploadBtn");
        const clipProgressWrap = clipPage.querySelector("#clipProgressWrap"), clipProgBar = clipPage.querySelector("#clipProgBar"), clipProgText = clipPage.querySelector("#clipProgText");
        const clipResultsWrap = clipPage.querySelector("#clipResultsWrap");

        clipFileInput.onchange = () => { clipFileName.textContent = clipFileInput.files[0]?.name || "未选择"; };

        clipUploadBtn.onclick = () => {
            if (!clipFileInput.files.length || !clipQueryInput.value.trim()) {
                return alert("请选择视频并输入检索文字！");
            }

            const formData = new FormData();
            formData.append("video", clipFileInput.files[0]);
            formData.append("query", clipQueryInput.value.trim());
            formData.append("top_n", clipTopNInput.value);
            formData.append("threshold", clipThresholdInput.value); // 【新增】添加阈值到表单

            clipProgressWrap.hidden = false;
            clipProgBar.value = 0;
            clipProgText.textContent = "上传中...";
            clipUploadBtn.disabled = true;
            clipResultsWrap.innerHTML = '';

            const xhr = new XMLHttpRequest();
            xhr.open("POST", "/clip/search", true);
            xhr.upload.onprogress = e => {
                if (e.lengthComputable) {
                    const pct = Math.round((e.loaded / e.total) * 100);
                    clipProgBar.value = pct;
                    if (pct === 100) {
                        clipProgText.textContent = "服务器处理中，请稍候...";
                    }
                }
            };
            xhr.responseType = "json";
            xhr.onload = () => {
                clipUploadBtn.disabled = false;
                clipProgressWrap.hidden = true;
                if (xhr.status !== 200) {
                    return alert(`错误: ${xhr.response?.detail || xhr.statusText}`);
                }
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
            // ... (此函数内容保持不变) ...
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

    // --- 页面3: 操作历史 (History) ---
    const historyPage = $("history-page");
    if (historyPage) {
        const historyList = historyPage.querySelector("#historyList");
        const clearHistoryBtn = historyPage.querySelector("#clearHistoryBtn");

        async function loadHistory() {
            try {
                historyList.innerHTML = '<p class="info">正在加载历史记录...</p>';
                const response = await fetch('/history');
                const data = await response.json();
                renderHistory(data.history);
            } catch (error) {
                historyList.innerHTML = '<p class="info">加载历史记录失败。</p>';
            }
        }

        clearHistoryBtn.onclick = async () => {
            if (!confirm("确定要清空所有历史记录吗？此操作不可恢复。")) return;
            try {
                await fetch('/history', { method: 'DELETE' });
                loadHistory();
            } catch (error) {
                alert("清空历史记录失败。");
            }
        };

        function renderHistory(historyData) {
            if (!historyData || historyData.length === 0) {
                historyList.innerHTML = '<p class="info">暂无历史记录。</p>';
                return;
            }
            historyList.innerHTML = '';
            historyData.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';
                let iconClass = 'fa-question-circle';
                let previewHTML = '';
                if (item.type === '全身行人检索') {
                    iconClass = 'fa-user-check';
                    previewHTML = `<a href="${item.result_url}" target="_blank"><video src="${item.result_url}" preload="metadata"></video></a>`;
                } else if (item.type === '图片内容检索') {
                    iconClass = 'fa-image';
                    previewHTML = `<a href="${item.result_url}" target="_blank"><img src="${item.result_url}" alt="检索结果"></a>`;
                } else if (item.type === '视频片段检索' && item.results?.length > 0) {
                    iconClass = 'fa-film';
                    previewHTML = `<a href="${item.results[0].video_url}" target="_blank"><video src="${item.results[0].video_url}" preload="metadata"></video></a>`;
                }
                div.innerHTML = `
                    <div class="history-item-icon"><i class="fas ${iconClass}"></i></div>
                    <div class="history-item-content">
                        <h5>${item.type}</h5>
                        <p>查询: <strong>${item.query}</strong></p>
                        <span class="timestamp">${item.timestamp}</span>
                    </div>
                    <div class="history-item-preview">${previewHTML}</div>
                `;
                historyList.appendChild(div);
            });
        }
    }
});
