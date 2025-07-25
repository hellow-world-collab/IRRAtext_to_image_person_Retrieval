// 文件名: script.js (最终完整修正版)
// 描述: 包含所有页面的功能，并修复了语法和作用域问题。

document.addEventListener('DOMContentLoaded', () => {

    const $ = id => document.getElementById(id);
    const ITEMS_PER_PAGE = 10; // 每页显示10条

    // --- 全局元素 ---
    const appContainer = $("app-container");
    const menuToggle = $("menu-toggle");
    const menuLinks = document.querySelectorAll('.menu-links li');
    const pages = document.querySelectorAll('.page-content');

    // --- 页面1: 全身行人检索 (Person Search) ---
    const personSearchPage = $("person-search-page");
    if (personSearchPage) {
        // 模式切换逻辑
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

        // 图片检索元素和逻辑
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

    // --- 页面2: 视频片段检索 (Clip Search) ---
    const clipPage = $("clip-search-page");
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

    // --- 页面3: 操作历史 (History) ---
    const historyPage = $("history-page");
    async function loadHistory(page = 1) {
        const historyTableBody = historyPage.querySelector("#historyTableBody");
        const paginationContainer = historyPage.querySelector("#pagination-container");
        if (!historyTableBody || !paginationContainer) return;

        try {
            historyTableBody.innerHTML = '<tr><td colspan="5" class="no-history-row">正在加载历史记录...</td></tr>';
            paginationContainer.innerHTML = '';

            const response = await fetch(`/history?page=${page}&limit=${ITEMS_PER_PAGE}`);
            if (!response.ok) throw new Error("无法从服务器获取历史记录。");

            const data = await response.json();
            renderHistory(data.items);
            renderPagination(data.page, data.pages);

        } catch (error) {
            historyTableBody.innerHTML = `<tr><td colspan="5" class="no-history-row">加载历史记录失败: ${error.message}</td></tr>`;
        }
    }

    function renderHistory(historyData) {
        const historyTableBody = historyPage.querySelector("#historyTableBody");
        if (!historyTableBody) return;

        if (!historyData || historyData.length === 0) {
            historyTableBody.innerHTML = '<tr class="no-history-row"><td colspan="5">暂无历史记录。</td></tr>';
            return;
        }

        historyTableBody.innerHTML = '';
        historyData.forEach(item => {
            const row = document.createElement('tr');
            let iconClass = 'fa-question-circle', previewHTML = '<span>无预览</span>';

            if (item.operation_type === '全身行人检索' && item.result_url) {
                iconClass = 'fa-user-check';
                previewHTML = `<a href="${item.result_url}" target="_blank"><video src="${item.result_url}" preload="metadata"></video></a>`;
            } else if (item.operation_type === '图片内容检索' && item.result_url) {
                iconClass = 'fa-image';
                previewHTML = `<a href="${item.result_url}" target="_blank"><img src="${item.result_url}" alt="检索结果"></a>`;
            } else if (item.operation_type === '视频片段检索') {
                iconClass = 'fa-film';
                try {
                    const details = JSON.parse(item.details || '{}');
                    const firstClip = details.results?.[0];
                    if(firstClip && firstClip.video_url) {
                        previewHTML = `<a href="${firstClip.video_url}" target="_blank"><video src="${firstClip.video_url}" preload="metadata"></video></a>`;
                    }
                } catch(e) { console.error("Error parsing history details:", e); }
            }

            row.innerHTML = `
                <td>${item.id}</td>
                <td><i class="fas ${iconClass}"></i> ${item.operation_type}</td>
                <td class="query-text">${item.query_text || 'N/A'}</td>
                <td class="preview-cell">${previewHTML}</td>
                <td>${item.timestamp}</td>
            `;
            historyTableBody.appendChild(row);
        });
    }

    function renderPagination(currentPage, totalPages) {
        const paginationContainer = historyPage.querySelector("#pagination-container");
        if (!paginationContainer) return;

        paginationContainer.innerHTML = '';
        if (totalPages <= 1) return;

        const prevButton = document.createElement('button');
        prevButton.innerHTML = `<i class="fas fa-angle-left"></i>`;
        prevButton.className = 'pagination-btn';
        prevButton.disabled = currentPage === 1;
        prevButton.onclick = () => loadHistory(currentPage - 1);
        paginationContainer.appendChild(prevButton);

        for (let i = 1; i <= totalPages; i++) {
            const pageButton = document.createElement('button');
            pageButton.textContent = i;
            pageButton.className = 'pagination-btn';
            if (i === currentPage) pageButton.classList.add('active');
            pageButton.onclick = () => loadHistory(i);
            paginationContainer.appendChild(pageButton);
        }

        const nextButton = document.createElement('button');
        nextButton.innerHTML = `<i class="fas fa-angle-right"></i>`;
        nextButton.className = 'pagination-btn';
        nextButton.disabled = currentPage === totalPages;
        nextButton.onclick = () => loadHistory(currentPage + 1);
        paginationContainer.appendChild(nextButton);
    }

    if (historyPage) {
        const clearHistoryBtn = historyPage.querySelector("#clearHistoryBtn");
        if(clearHistoryBtn) {
            clearHistoryBtn.onclick = async () => {
                if (!confirm("确定要清空所有历史记录吗？")) return;
                try {
                    await fetch('/history', { method: 'DELETE' });
                    loadHistory(1);
                } catch (error) {
                    alert("清空历史记录失败。");
                }
            };
        }
    }

    // --- 全局事件监听 ---
    if(menuToggle) menuToggle.addEventListener('click', () => {
        appContainer.classList.toggle('sidebar-open');
    });

    menuLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageId = link.getAttribute('data-page');
            pages.forEach(page => page.classList.toggle('hidden', page.id !== pageId));
            menuLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            if (pageId === 'history-page') {
                loadHistory(1);
            }
            if (window.innerWidth < 768) appContainer.classList.remove('sidebar-open');
        });
    });
});