// 文件名: web/pages/dashboard/dashboard.js (最终清洁版 - 已修正)
(function() {
    let typeChart, activityChart;

    async function loadDashboardData() {
        try {
            const [statsRes, topQueriesRes, historyRes] = await Promise.all([
                fetch('/dashboard/stats'),
                fetch('/dashboard/top-queries'),
                fetch('/history?page=1&limit=5')
            ]);

            if (!statsRes.ok || !topQueriesRes.ok || !historyRes.ok) {
                throw new Error("一个或多个数据接口请求失败");
            }

            const statsData = await statsRes.json();
            const topQueriesData = await topQueriesRes.json();
            const historyData = await historyRes.json();

            document.getElementById('total-searches').textContent = statsData.total_searches;
            document.getElementById('today-searches').textContent = statsData.today_searches;

            renderTypeDistributionChart(statsData.type_distribution);
            renderDailyActivityChart(statsData.daily_activity);
            renderWordCloud(topQueriesData);
            renderRecentHistory(historyData.items);

        } catch (error) {
            console.error("加载仪表盘数据失败:", error);
            const pageContainer = document.getElementById('page-container');
            if (pageContainer) {
                pageContainer.innerHTML = `<div class="card" style="text-align:center;"><h2>仪表盘数据加载失败</h2><p>${error.message}</p></div>`;
            }
        }
    }

    function renderTypeDistributionChart(data) {
        const ctx = document.getElementById('type-distribution-chart')?.getContext('2d');
        if (!ctx) return;
        if (typeChart) typeChart.destroy();
        Chart.register(ChartDataLabels);
        typeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.map(d => d.operation_type),
                datasets: [{ data: data.map(d => d.count), backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0'] }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' },
                    datalabels: {
                        formatter: (value, ctx) => {
                            let sum = ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                            return sum === 0 ? '0%' : (value * 100 / sum).toFixed(1) + "%";
                        },
                        color: '#fff',
                    }
                }
            }
        });
    }

    function renderDailyActivityChart(data) {
        const ctx = document.getElementById('daily-activity-chart')?.getContext('2d');
        if (!ctx) return;
        if (activityChart) activityChart.destroy();
        const labels = Object.keys(data).sort();
        const values = labels.map(label => data[label] || 0);
        activityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{ label: '每日检索量', data: values, backgroundColor: 'rgba(54, 162, 235, 0.6)' }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } }
        });
    }

    function renderWordCloud(data) {
        const canvas = document.getElementById('wordcloud-canvas');
        if (!canvas) return;

        if (typeof WordCloud === 'undefined') {
            console.error("WordCloud library is not loaded yet!");
            canvas.getContext("2d").fillText("词云库加载失败", 10, 50);
            return;
        }

        if (!data || !data.length) {
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.font = "20px Inter";
            ctx.fillStyle = "#999";
            ctx.textAlign = "center";
            ctx.fillText("暂无热门搜索词", canvas.width / 2, canvas.height / 2);
            return;
        }

        const list = data.map(item => [item.query_text, item.count * 10]);
        WordCloud(canvas, { list: list, gridSize: 10, weightFactor: 2, fontFamily: 'Inter', color: 'random-dark' });
    }

    function renderRecentHistory(items) {
        const listEl = document.getElementById('recent-history-list');
        if (!listEl) return;
        listEl.innerHTML = '';
        if (!items || !items.length) {
            listEl.innerHTML = '<li>暂无操作记录</li>';
            return;
        }
        items.forEach(item => {
            const li = document.createElement('li');
            let iconClass = 'fa-question-circle';
            if (item.operation_type === '全身行人检索') iconClass = 'fa-user-check';
            if (item.operation_type === '图片内容检索') iconClass = 'fa-image';
            if (item.operation_type === '视频片段检索') iconClass = 'fa-film';
            li.innerHTML = `<i class="fas ${iconClass} icon"></i><span><strong>${item.operation_type}:</strong> ${item.query_text}</span><span class="timestamp">${item.timestamp}</span>`;
            listEl.appendChild(li);
        });
    }

    loadDashboardData();
})();
