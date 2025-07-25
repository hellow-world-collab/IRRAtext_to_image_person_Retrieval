// 文件名: web/pages/history/history.js (已修正)
// 描述: 修正了致命的语法错误，并重构了代码逻辑，使其能正确工作

(function() {
    // 使用IIFE（立即执行函数表达式）创建一个私有作用域，防止变量冲突

    // --- 1. 定义变量 ---
    const ITEMS_PER_PAGE = 10;
    let currentPage = 1;

    // --- 2. 获取页面元素 ---
    // 我们在页面加载后立即获取所有需要的元素
    const historyTableBody = document.querySelector("#historyTableBody");
    const paginationContainer = document.querySelector("#pagination-container");
    const selectAllCheckbox = document.querySelector("#selectAllCheckbox");
    const clearHistoryBtn = document.querySelector("#clearHistoryBtn");
    const deleteSelectedBtn = document.querySelector("#deleteSelectedBtn");

    // 安全检查，如果关键元素不存在，则不执行任何操作
    if (!historyTableBody || !paginationContainer || !clearHistoryBtn) {
        console.error("历史记录页面的关键HTML元素未找到！");
        return;
    }

    // --- 3. 定义功能函数 ---

    async function loadHistory(page = 1) {
        currentPage = page;
        try {
            historyTableBody.innerHTML = '<tr><td colspan="6" class="no-history-row">正在加载历史记录...</td></tr>';
            paginationContainer.innerHTML = '';

            const response = await fetch(`/history?page=${page}&limit=${ITEMS_PER_PAGE}`);
            if (!response.ok) throw new Error("无法从服务器获取历史记录。");

            const data = await response.json();
            renderHistory(data.items);
            renderPagination(data.page, data.pages);
            if(selectAllCheckbox) selectAllCheckbox.checked = false;
        } catch (error) {
            historyTableBody.innerHTML = `<tr><td colspan="6" class="no-history-row">加载历史记录失败: ${error.message}</td></tr>`;
        }
    }

    function renderHistory(historyData) {
        if (!historyData || historyData.length === 0) {
            historyTableBody.innerHTML = '<tr class="no-history-row"><td colspan="6">暂无历史记录。</td></tr>';
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
                } catch(e) { console.error("解析历史详情失败:", e); }
            }
            row.innerHTML = `
                <td class="checkbox-cell"><input type="checkbox" class="history-checkbox" value="${item.id}"></td>
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

    // --- 4. 绑定所有事件监听器 ---

    clearHistoryBtn.onclick = async () => {
        if (!confirm("确定要清空所有历史记录吗？此操作不可恢复！")) return;
        try {
            const response = await fetch('/history', { method: 'DELETE' });
            if(!response.ok) throw new Error("服务器响应错误");
            alert("所有历史记录已清空。");
            loadHistory(1);
        } catch (error) {
            alert(`清空历史记录失败: ${error.message}`);
        }
    };

    deleteSelectedBtn.onclick = async () => {
        const selectedCheckboxes = historyTableBody.querySelectorAll('.history-checkbox:checked');
        const idsToDelete = Array.from(selectedCheckboxes).map(cb => cb.value);

        if (idsToDelete.length === 0) {
            return alert("请至少选择一项要删除的记录。");
        }
        if (!confirm(`确定要删除选中的 ${idsToDelete.length} 条记录吗？`)) return;

        try {
            const response = await fetch('/history/selected', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids: idsToDelete })
            });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "删除失败");
            }
            const result = await response.json();
            alert(result.message);
            if(selectedCheckboxes.length === historyTableBody.children.length && currentPage > 1){
                 loadHistory(currentPage - 1);
            } else {
                 loadHistory(currentPage);
            }
        } catch (error) {
            alert(`删除失败: ${error.message}`);
        }
    };

    if(selectAllCheckbox) {
        selectAllCheckbox.onchange = (e) => {
            const isChecked = e.target.checked;
            const checkboxes = historyTableBody.querySelectorAll('.history-checkbox');
            checkboxes.forEach(cb => cb.checked = isChecked);
        };
    }

    // --- 5. 初始加载 ---
    loadHistory(1);

})(); // IIFE 结束