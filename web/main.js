// 文件名: web/main.js (最终稳定版 - 已修正)
document.addEventListener('DOMContentLoaded', () => {
    const pageContainer = document.getElementById('page-container');
    const menuLinks = document.querySelectorAll('.menu-links li');
    const appContainer = document.getElementById("app-container");
    const menuToggle = document.getElementById("menu-toggle");

    const loadedStyles = new Set();

    async function loadPage(pageName) {
        try {
            const htmlResponse = await fetch(`pages/${pageName}/${pageName}.html`);
            if (!htmlResponse.ok) throw new Error(`无法加载 ${pageName}.html`);
            pageContainer.innerHTML = await htmlResponse.text();

            const scriptPath = `pages/${pageName}/${pageName}.js?t=${new Date().getTime()}`;
            const scriptElement = document.createElement('script');
            scriptElement.src = scriptPath;
            document.body.appendChild(scriptElement).parentNode.removeChild(scriptElement);

            const cssPath = `pages/${pageName}/${pageName}.css`;
            if (!loadedStyles.has(cssPath)) {
                const cssResponse = await fetch(cssPath);
                if (cssResponse.ok) {
                    const linkElement = document.createElement('link');
                    linkElement.rel = 'stylesheet';
                    linkElement.href = cssPath;
                    document.head.appendChild(linkElement);
                    loadedStyles.add(cssPath);
                }
            }
        } catch (error) {
            pageContainer.innerHTML = `<div class="card" style="text-align:center;"><h2>页面加载失败</h2><p>${error.message}</p></div>`;
            console.error(error);
        }
    }

    menuLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageName = link.getAttribute('data-page');
            menuLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            loadPage(pageName);
            if (window.innerWidth < 768) {
                appContainer.classList.remove('sidebar-open');
            }
        });
    });

    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            appContainer.classList.toggle('sidebar-open');
        });
    }

    loadPage('dashboard');
});