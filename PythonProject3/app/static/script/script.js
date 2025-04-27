// 在 script.js 文件中添加以下代码
document.addEventListener('DOMContentLoaded', function () {
    const browseBtn = document.getElementById('browseBtn');
    const directoryInput = document.getElementById('directoryInput');
    const pathInput = document.getElementById('pathInput');
    const trainSVMBtn = document.getElementById('trainSVMBtn');
    const model1Btn = document.getElementById('model1Btn');
    const clearAllBtn = document.getElementById('clearAll');
document.getElementById('exportResultBtn').addEventListener('click', function() {
    window.location.href = '/download_results';
});


const importAllBtn = document.getElementById('importAllBtn');

if (importAllBtn && directoryInput) {
    importAllBtn.addEventListener('click', async function () {
        // 禁用按钮防止重复点击
        importAllBtn.disabled = true;
        // importAllBtn.textContent = '处理中...';

        try {
            const files = directoryInput.files;

            if (!files || files.length === 0) {
                showToast('请先选择文件', 'warning');
                return;
            }

            // 显示处理进度
            const progress = document.createElement('div');
            progress.className = 'import-progress';
            importAllBtn.after(progress);

            // 使用Promise.all并行处理（如果操作是独立的）
            const processPromises = Array.from(files).map((file, index) => {
                return new Promise((resolve) => {
                    try {
                        const path = file.webkitRelativePath || file.name; // 兼容性处理

                        // 更新进度
                        // progress.textContent = `处理中 ${index + 1}/${files.length}: ${file.name}`;

                        // 执行处理函数
                        addToPreprocessQueue(file, path);
                        handleHDRSelection(file, files);

                        resolve(true);
                    } catch (error) {
                        console.error(`处理文件 ${file.name} 时出错:`, error);
                        resolve(false);
                    }
                });
            });

            // 等待所有文件处理完成
            const results = await Promise.all(processPromises);
            const successCount = results.filter(Boolean).length;

            // 显示完成状态
            showToast(`成功处理 ${files.length} 个文件`, 'success');
            // progress.textContent = `完成 ${successCount}/${files.length}`;

            // 5 秒后自动移除进度显示
            setTimeout(() => {
                progress.remove();
            }, 3000);

        } catch (error) {
            console.error('批量导入出错:', error);
            showToast('导入过程中出错', 'error');
        } finally {
            // 恢复按钮状态
            importAllBtn.disabled = false;
            importAllBtn.textContent = '导入全部';
        }
    });
} else {
    console.error('未找到导入按钮或文件输入元素');
}

// 简单的提示函数
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}
// 简单的提示函数
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}
    // 点击浏览按钮时，触发文件选择框
    browseBtn.addEventListener('click', function () {
        directoryInput.click();
    });

    // 当用户选择目录后，处理选择的文件
    directoryInput.addEventListener('change', function () {
        const files = directoryInput.files;
        if (files.length > 0) {
            // 这里可以显示选择的路径，简单示例显示第一个文件的相对路径
            pathInput.value = files[0].webkitRelativePath.split('/')[0];

            // 自动添加 -background.HDR 文件
            const dt = new DataTransfer();
            const allFiles = Array.from(files);

            // 构建映射：相对路径 -> 文件对象
            const fileMap = {};
            for (const file of allFiles) {
                fileMap[file.webkitRelativePath] = file;
            }

            // 添加主文件与匹配的 background 文件
            for (const file of allFiles) {
                const relPath = file.webkitRelativePath;

                // 只处理 .HDR 且不包含 -background 的文件
                if (relPath.endsWith('.HDR') && !relPath.includes('-background')) {
                    const dir = relPath.substring(0, relPath.lastIndexOf('/') + 1);
                    const base = relPath.substring(relPath.lastIndexOf('/') + 1).replace('.HDR', '');
                    const bgName = `${dir}${base}-background.HDR`;

                    if (fileMap[bgName]) {
                        // 自动添加背景文件
                        dt.items.add(fileMap[bgName]);
                        console.log(`✅ 自动添加背景文件: ${bgName}`);
                    } else {
                        console.warn(`⚠️ 找不到背景文件: ${bgName}`);
                        alert(`⚠️ 警告：找不到背景文件: ${bgName}`);
                    }
                }

                // 添加原始文件
                dt.items.add(file);
            }

            // 更新 directoryInput 的文件内容
            directoryInput.files = dt.files;

            // 构建文件树
            buildFileTree(directoryInput.files);
        }
    });

    // 点击清空表单按钮
    clearAllBtn.addEventListener('click', function () {
        // 清空表单
        clearForm();
    });


    // 点击大肠杆菌预测按钮
    trainSVMBtn.addEventListener('click', function () {
        processAndPredict('ecoli');
    });

    // 点击金黄色葡萄球菌预测按钮
    model1Btn.addEventListener('click', function () {
        processAndPredict('s_aureus');
    });

    let tabs = [];
    let activeTabId = null;
    let preprocessQueue = []; // 预处理队列

    // 创建 toast 提示框
    const toast = document.createElement('div');
    toast.className = 'toast';
    document.body.appendChild(toast);

    // 显示 toast 提示框
    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000); // 3 秒后自动消失
    }

    // ================== 目录结构相关 ==================

    // 切换文件夹展开和收起
    function toggleFolder(event) {
        event.stopPropagation();
        const folder = event.target.closest('.folder');
        if (!folder) return;

        folder.classList.toggle('open');
        const subItems = folder.nextElementSibling;
        if (subItems) subItems.classList.toggle('hidden');
    }

    function clearForm() {
    // 清空预处理队列
    preprocessQueue = [];

    // 清空表格
    const tableBody = document.getElementById('tableBody');
    tableBody.innerHTML = '';

    // 清空文件输入
    directoryInput.value = '';

    // 清空路径输入
    pathInput.value = '';

    // 清空temp文件夹
    clearTempDirectory();

    // 清空地址树
    const sidebar = document.getElementById('sidebar');
    sidebar.innerHTML = '';

    // 清空result文件夹
    clearResultDirectory();

    // 显示提示信息
    showToast('表单已清空');
    }


    // 构建文件树
    function buildFileTree(files) {
        const sidebar = document.getElementById('sidebar');
        sidebar.innerHTML = '';
        const fileTree = {};

        Array.from(files).forEach(file => {
            const pathParts = file.webkitRelativePath.split('/');
            const ext = pathParts[pathParts.length - 1].split('.').pop().toLowerCase();
            // 只处理 .HDR 文件
            if (ext === 'hdr') {
                let currentLevel = fileTree;
                pathParts.forEach((part, index) => {
                    if (!currentLevel[part]) {
                        // 保存完整路径
                        currentLevel[part] = index === pathParts.length - 1 ? { $file: file, path: file.webkitRelativePath, fullPath: file.webkitRelativePath } : {};
                    }
                    currentLevel = currentLevel[part];
                });
            }
        });

        function createTreeItem(name, content) {
            const isFile = content.$file;
            const item = document.createElement('div');
            item.className = isFile ? 'file-item' : 'folder';
            item.textContent = name;

            if (isFile) {
                setupSingleDoubleClick(item, content.$file, content.fullPath);
            } else {
                item.addEventListener('click', toggleFolder);
            }

            return item;
        }

        function renderTree(node, container) {
            Object.entries(node).forEach(([name, content]) => {
                if (name === '$file') return;

                const item = createTreeItem(name, content);
                container.appendChild(item);

                if (!content.$file) {
                    const subContainer = document.createElement('div');
                    subContainer.className = 'sub-items hidden';
                    renderTree(content, subContainer);
                    container.appendChild(subContainer);
                }
            });
        }

        renderTree(fileTree, sidebar);
    }

    // 设置单双击事件
    function setupSingleDoubleClick(element, file, path) {
        let clickTimer = null;
        let clickCount = 0;

        element.addEventListener('click', function (event) {
            clickCount++;
            if (clickCount === 1) {
                clickTimer = setTimeout(() => {
                    previewFile(file);
                    clickCount = 0;
                }, 300); // 300ms 作为区分单双击的时间阈值
            } else if (clickCount === 2) {
                clearTimeout(clickTimer);
                addToPreprocessQueue(file, path);
                handleHDRSelection(file, directoryInput.files); // 调用 handleHDRSelection 函数
                clickCount = 0;
            }
        });
    }

    // 将文件添加到预处理队列
    async function addToPreprocessQueue(file, path) {
        const relatedFiles = findRelatedFiles(file, path);
        const mainFile = relatedFiles.find(f => f.name === file.name);

        if (!preprocessQueue.some(f => f.name === mainFile.name)) {
            // 保存文件到 app\temp\orign 和 app\temp\background 文件夹
            const originDir = 'app\\temp\\orign';
            const backgroundDir = 'app\\temp\\background';
            for (const relatedFile of relatedFiles) {
                const reader = new FileReader();
                reader.readAsArrayBuffer(relatedFile);
                reader.onload = async () => {
                    const buffer = reader.result;
                    const fs = window.require('fs');
                    const path = window.require('path');
                    let filePath;
                    if (relatedFile.name === mainFile.name) {
                        filePath = path.join(originDir, relatedFile.name);
                    } else {
                        filePath = path.join(backgroundDir, relatedFile.name);
                    }
                    try {
                        await fs.promises.writeFile(filePath, Buffer.from(buffer));
                    } catch (error) {
                        console.error(`Error saving file ${relatedFile.name}: ${error}`);
                    }
                };
            }

            preprocessQueue.push(...relatedFiles);
            showToast(`${mainFile.name} 及其相关文件已添加到预处理队列。`);
            // 仅将主文件添加到表格
            const tableBody = document.getElementById('tableBody');
            const row = tableBody.insertRow();
            const cell1 = row.insertCell(0);
            cell1.textContent = mainFile.name;
            const cell2 = row.insertCell(1);
            cell2.textContent = '';
            const cell3 = row.insertCell(2);
            cell3.textContent = '';
            const cell4 = row.insertCell(3);
            const closeButton = document.createElement('span');
            closeButton.textContent = '×';
            closeButton.style.cursor = 'pointer';
            closeButton.addEventListener('click', () => removeFromPreprocessQueue(mainFile.name));
            cell4.appendChild(closeButton);
        } else {
            showToast(`${mainFile.name} 已在预处理队列中。`);
        }
    }

    // 自动检索对应的背景文件
    function findRelatedFiles(file, path) {
        const files = directoryInput.files;
        const baseName = file.name.replace(/\.[^/.]+$/, '');
        const relatedFiles = [file];
        const backgroundFileName = baseName + '-background.HDR';
        const backgroundFilePath = path.replace(file.name, backgroundFileName);
        const backgroundFile = Array.from(files).find(f => f.webkitRelativePath === backgroundFilePath);
        if (backgroundFile) {
            relatedFiles.push(backgroundFile);
        }

            // 查找与HDR文件基本名称相同的raw文件
        const rawFileName = baseName + '.raw';
        const rawFilePath = path.replace(file.name, rawFileName);
        const rawFile = Array.from(files).find(f => f.webkitRelativePath === rawFilePath);
        if (rawFile) {
            relatedFiles.push(rawFile);
        }

        const backgroudrawFileName = baseName + '-background' + '.raw';
        const backgroudrawFilePath = path.replace(file.name, backgroudrawFileName);
        const backgroudrawFile = Array.from(files).find(f => f.webkitRelativePath === backgroudrawFilePath);
        if (backgroudrawFile) {
            relatedFiles.push(backgroudrawFile);
        }

        return relatedFiles;
    }

    // 从预处理队列中移除文件
    function removeFromPreprocessQueue(fileName) {
        const index = preprocessQueue.findIndex(f => f.name === fileName);
        if (index !== -1) {
            // 移除相关文件
            const baseName = fileName.replace(/\.[^/.]+$/, '');
            preprocessQueue = preprocessQueue.filter(f => !f.name.startsWith(baseName));
            const tableBody = document.getElementById('tableBody');
            for (let i = 0; i < tableBody.rows.length; i++) {
                if (tableBody.rows[i].cells[0].textContent === fileName) {
                    tableBody.deleteRow(i);
                    break;
                }
            }
            showToast(`${fileName} 及其相关文件已从预处理队列中移除。`);
        }
    }


    // 预处理并预测
    function processAndPredict(modelType) {
        const form = document.getElementById('classificationForm');
        const fileInput = form.querySelector('input[type="file"]');

        // 使用 DataTransfer 对象来模拟 FileList
        const dataTransfer = new DataTransfer();
        const originalPaths = [];
        preprocessQueue.forEach(file => {
            dataTransfer.items.add(file);
            originalPaths.push(file.webkitRelativePath);
        });
        fileInput.files = dataTransfer.files;

        // 添加原始路径信息到表单
        const originalPathInput = document.createElement('input');
        originalPathInput.type = 'hidden';
        originalPathInput.name = 'original_path';
        originalPathInput.value = originalPaths.join(',');
        form.appendChild(originalPathInput);

        const xhr = new XMLHttpRequest();
        // 确保URL与Flask应用的路由一致
        xhr.open('POST', modelType === 'ecoli' ? '/classify_ecoli' : '/classify_s_aureus', true);
        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4 && xhr.status === 200) {
                const results = JSON.parse(xhr.responseText);
                updateTable(results);
                clearTempDirectory(); // 添加清空temp文件夹的调用
            }
        };
        xhr.send(new FormData(form));
    }

    // 清空temp文件夹
    function clearTempDirectory() {
        const fs = window.require('fs');
        const path = window.require('path');
        const tempDir = path.join('app', 'temp');

        fs.readdir(tempDir, (err, files) => {
            if (err) throw err;

            for (const file of files) {
                fs.unlink(path.join(tempDir, file), err => {
                    if (err) throw err;
                });
            }
        });
    }

    // 清空result文件夹
    function clearResultDirectory() {
        const fs = window.require('fs');
        const path = window.require('path');
        const resultDir = path.join('app', 'result');

        fs.readdir(resultDir, (err, files) => {
            if (err) throw err;

            for (const file of files) {
                fs.unlink(path.join(resultDir, file), err => {
                    if (err) throw err;
                });
            }
        });
    }


    // 预处理并预测
    // function processAndPredict(modelType) {
    //     const form = document.getElementById('classificationForm');
    //     const fileInput = form.querySelector('input[type="file"]');
    //
    //     // 使用 DataTransfer 对象来模拟 FileList
    //     const dataTransfer = new DataTransfer();
    //     const originalPaths = [];
    //     preprocessQueue.forEach(file => {
    //         dataTransfer.items.add(file);
    //         originalPaths.push(file.webkitRelativePath);
    //     });
    //     fileInput.files = dataTransfer.files;
    //
    //     // 添加原始路径信息到表单
    //     const originalPathInput = document.createElement('input');
    //     originalPathInput.type = 'hidden';
    //     originalPathInput.name = 'original_path';
    //     originalPathInput.value = originalPaths.join(',');
    //     form.appendChild(originalPathInput);
    //
    //     const xhr = new XMLHttpRequest();
    //     // 确保URL与Flask应用的路由一致
    //     xhr.open('POST', modelType === 'ecoli' ? '/classify_ecoli' : '/classify_s_aureus', true);
    //     xhr.onreadystatechange = function () {
    //         if (xhr.readyState === 4 && xhr.status === 200) {
    //             const results = JSON.parse(xhr.responseText);
    //             updateTable(results);
    //         }
    //     };
    //     xhr.send(new FormData(form));
    // }

    // 更新表格
    // 更新表格
    function updateTable(results) {
        const tableBody = document.getElementById('tableBody');
        results.forEach(result => {
            for (let i = 0; i < tableBody.rows.length; i++) {
                if (tableBody.rows[i].cells[0].textContent === result.filename) {
                    const cell2 = tableBody.rows[i].cells[1];
                    const cell3 = tableBody.rows[i].cells[2];
                    if (result.prediction === 0) {
                        cell2.textContent = `✔ ${(result.probability * 100).toFixed(2)}%`;
                        cell3.textContent = '';
                    } else if (result.prediction === 1) {
                        cell2.textContent = '';
                        cell3.textContent = `✔ ${(result.probability * 100).toFixed(2)}%`;
                    }
                    break;
                }
            }
        });
    }

    // function updateTable(results) {
    //     const tableBody = document.getElementById('tableBody');
    //     results.forEach(result => {
    //         for (let i = 0; i < tableBody.rows.length; i++) {
    //             if (tableBody.rows[i].cells[0].textContent === result.filename) {
    //                 const cell2 = tableBody.rows[i].cells[1];
    //                 const cell3 = tableBody.rows[i].cells[2];
    //                 const positiveProb = result.probabilities['positive'] || 0;
    //                 const negativeProb = result.probabilities['negative'] || 0;
    //                 if (positiveProb > negativeProb) {
    //                     cell2.textContent = `✔ ${(positiveProb * 100).toFixed(2)}%`;
    //                     cell3.textContent = '';
    //                 } else {
    //                     cell2.textContent = '';
    //                     cell3.textContent = `✔ ${(negativeProb * 100).toFixed(2)}%`;
    //                 }
    //                 break;
    //             }
    //         }
    //     });
    // }

    // ================== 预览相关 ==================

    // 创建标签
    function createTab(file, content) {
        const tabId = `tab-${Date.now()}`;
        return { id: tabId, file, content, element: null, button: null };
    }

    // 添加标签
    function addTab(tab) {
        const tabButton = document.createElement('div');
        tabButton.className = 'tab-button';
        tabButton.innerHTML = `
            <span class="tab-filename">${tab.file.name}</span>
            <span class="tab-close">×</span>
        `;
        tabButton.dataset.tabId = tab.id;

        // 监听单击关闭按钮
        tabButton.querySelector('.tab-close').addEventListener('click', (e) => {
            e.stopPropagation();
            closeTab(tab.id);
        });

        // 监听双击关闭标签页
        tabButton.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            closeTab(tab.id);
        });

        tabButton.addEventListener('click', () => activateTab(tab.id));

        const contentPanel = document.createElement('div');
        contentPanel.className = 'content-panel';
        contentPanel.appendChild(tab.content);

        tab.button = tabButton;
        tab.element = contentPanel;

        document.getElementById('previewTabBar').appendChild(tabButton);
        document.getElementById('previewArea').appendChild(contentPanel);
        tabs.push(tab);

        updateFilename(tabButton);

        const resizeObserver = new ResizeObserver(() => updateFilename(tabButton));
        resizeObserver.observe(tabButton);
        tabButton._resizeObserver = resizeObserver;

        updateTabWidths();  // 调用更新宽度的函数
    }

    // 激活标签
    function activateTab(tabId) {
        tabs.forEach(tab => {
            const isActive = tab.id === tabId;
            tab.button.classList.toggle('active', isActive);
            tab.element.style.display = isActive ? 'block' : 'none';
        });
        activeTabId = tabId;
    }

    // 关闭标签
    function closeTab(tabId) {
        const index = tabs.findIndex(t => t.id === tabId);
        if (index === -1) return;

        const [tab] = tabs.splice(index, 1);
        tab.button.remove();
        tab.element.remove();

        if (tabId === activeTabId && tabs.length > 0) activateTab(tabs[0].id);
        if (tab.button._resizeObserver) tab.button._resizeObserver.disconnect();
    }

    // 更新文件名
    function updateFilename(tabButton) {
        const filenameSpan = tabButton.querySelector('.tab-filename');
        const containerWidth = tabButton.offsetWidth - 25;
        const fullFilename = filenameSpan.dataset.fullFilename || filenameSpan.textContent;

        // 保存完整文件名
        if (!filenameSpan.dataset.fullFilename) {
            filenameSpan.dataset.fullFilename = fullFilename;
        }

        // 计算文件名的宽度
        const tempSpan = document.createElement('span');
        tempSpan.textContent = fullFilename;
        tempSpan.style.visibility = 'hidden';
        tempSpan.style.position = 'absolute';
        document.body.appendChild(tempSpan);
        const filenameWidth = tempSpan.offsetWidth;
        document.body.removeChild(tempSpan);

        // 如果文件名宽度超过容器宽度，则进行截断
        if (filenameWidth > containerWidth) {
            let truncatedFilename = fullFilename;
            while (true) {
                const tempSpan = document.createElement('span');
                tempSpan.textContent = truncatedFilename + '...';
                tempSpan.style.visibility = 'hidden';
                tempSpan.style.position = 'absolute';
                document.body.appendChild(tempSpan);
                const truncatedWidth = tempSpan.offsetWidth;
                document.body.removeChild(tempSpan);

                if (truncatedWidth <= containerWidth) {
                    filenameSpan.textContent = truncatedFilename + '...';
                    break;
                }
                truncatedFilename = truncatedFilename.slice(0, -1);
            }
        } else {
            filenameSpan.textContent = fullFilename;
        }
    }

    // 更新标签宽度
    function updateTabWidths() {
        const tabBar = document.getElementById('previewTabBar');
        const tabButtons = tabBar.querySelectorAll('.tab-button');
        const totalWidth = tabBar.offsetWidth;
        const tabCount = tabButtons.length;
        const tabWidth = Math.floor(totalWidth / tabCount);

        tabButtons.forEach(button => {
            button.style.width = `${tabWidth}px`;
        });
    }

    // 预览文件
    function previewFile(file) {
        // 这里可以添加具体的预览逻辑，比如显示文件内容等
        console.log(`Previewing file: ${file.name}`);
    }

    // 模拟 FileList 工具
    function fileListFrom(files) {
        const dataTransfer = new DataTransfer();
        for (const file of files) {
            dataTransfer.items.add(file);
        }
        return dataTransfer.files; // 补充逻辑，返回模拟的 FileList
    }
}); // 补充 document.addEventListener 的右花括号
