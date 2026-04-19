const { app, BrowserWindow, Menu, dialog, ipcMain } = require('electron');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');

let mainWindow;
let pythonProcess;

// Wait for the FastAPI server to be ready
function waitForServer(url, timeout = 30000) {
    return new Promise((resolve, reject) => {
        const start = Date.now();
        const interval = setInterval(() => {
            if (Date.now() - start > timeout) {
                clearInterval(interval);
                reject(new Error('Timeout waiting for Python server'));
                return;
            }

            http.get(url, (res) => {
                if (res.statusCode === 200) {
                    clearInterval(interval);
                    resolve();
                }
            }).on('error', () => {
                // Ignore connection refused errors and keep trying
            });
        }, 500);
    });
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1440,
        height: 900,
        minWidth: 960,
        minHeight: 640,
        title: 'NeuroView MRI',
        icon: path.join(__dirname, 'app', 'static', 'icon.png'),
        backgroundColor: '#0a0e14',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        }
    });

    // ── Build the full native application menu ──────────────────────────
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Scan Files…',
                    accelerator: 'CmdOrCtrl+O',
                    click: async () => {
                        const result = await dialog.showOpenDialog(mainWindow, {
                            title: 'Open NIfTI Scan Files',
                            filters: [{ name: 'NIfTI Files', extensions: ['nii', 'nii.gz', 'gz'] }],
                            properties: ['openFile', 'multiSelections']
                        });
                        if (!result.canceled && result.filePaths.length) {
                            mainWindow.webContents.send('menu-open-files', result.filePaths);
                        }
                    }
                },
                { type: 'separator' },
                {
                    label: 'Reset Workspace',
                    accelerator: 'CmdOrCtrl+R',
                    click: () => mainWindow.webContents.send('menu-reset')
                },
                { type: 'separator' },
                {
                    label: 'Export Results…',
                    accelerator: 'CmdOrCtrl+E',
                    click: () => mainWindow.webContents.send('menu-export')
                },
                { type: 'separator' },
                { role: 'quit', accelerator: 'Alt+F4' }
            ]
        },
        {
            label: 'Edit',
            submenu: [
                {
                    label: 'Copy Slice Image',
                    accelerator: 'CmdOrCtrl+C',
                    click: () => mainWindow.webContents.send('menu-copy-slice')
                },
                { type: 'separator' },
                {
                    label: 'Preferences…',
                    accelerator: 'CmdOrCtrl+,',
                    click: () => mainWindow.webContents.send('menu-preferences')
                }
            ]
        },
        {
            label: 'View',
            submenu: [
                {
                    label: 'Toggle Sidebar',
                    accelerator: 'CmdOrCtrl+B',
                    click: () => mainWindow.webContents.send('menu-toggle-sidebar')
                },
                {
                    label: 'Toggle Status Bar',
                    accelerator: 'CmdOrCtrl+J',
                    click: () => mainWindow.webContents.send('menu-toggle-statusbar')
                },
                { type: 'separator' },
                { role: 'zoomIn', accelerator: 'CmdOrCtrl+=' },
                { role: 'zoomOut', accelerator: 'CmdOrCtrl+-' },
                { role: 'resetZoom', accelerator: 'CmdOrCtrl+0' },
                { type: 'separator' },
                { role: 'togglefullscreen' },
                { type: 'separator' },
                { role: 'toggleDevTools', accelerator: 'F12' }
            ]
        },
        {
            label: 'Tools',
            submenu: [
                {
                    label: 'Run Analysis',
                    accelerator: 'CmdOrCtrl+Shift+A',
                    click: () => mainWindow.webContents.send('menu-run-analysis')
                },
                { type: 'separator' },
                {
                    label: 'Axis',
                    submenu: [
                        {
                            label: 'Axial',
                            accelerator: 'Alt+1',
                            click: () => mainWindow.webContents.send('menu-change-axis', 'axial')
                        },
                        {
                            label: 'Coronal',
                            accelerator: 'Alt+2',
                            click: () => mainWindow.webContents.send('menu-change-axis', 'coronal')
                        },
                        {
                            label: 'Sagittal',
                            accelerator: 'Alt+3',
                            click: () => mainWindow.webContents.send('menu-change-axis', 'sagittal')
                        }
                    ]
                },
                { type: 'separator' },
                {
                    label: 'Toggle Tumor Overlay',
                    accelerator: 'CmdOrCtrl+T',
                    click: () => mainWindow.webContents.send('menu-toggle-overlay')
                },
                { type: 'separator' },
                {
                    label: 'Reload Backend',
                    click: () => { mainWindow.reload(); }
                }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'Documentation',
                    click: () => mainWindow.webContents.send('menu-docs')
                },
                {
                    label: 'Keyboard Shortcuts',
                    accelerator: 'CmdOrCtrl+/',
                    click: () => mainWindow.webContents.send('menu-shortcuts')
                },
                { type: 'separator' },
                {
                    label: 'About NeuroView MRI',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About NeuroView MRI',
                            message: 'NeuroView MRI',
                            detail: 'Version 1.0.0\n\nBrain tumor detection and visualization.\nPowered by PyTorch deep learning.\n\n© 2026 NeuroView Team',
                            buttons: ['OK']
                        });
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);

    mainWindow.loadURL('http://127.0.0.1:8000');

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.on('ready', async () => {
    console.log('Starting Python PyTorch backend...');

    if (app.isPackaged) {
        // In production, spawn the bundled executable
        const backendExe = path.join(process.resourcesPath, 'neuroview_backend', 'neuroview_backend.exe');
        pythonProcess = spawn(backendExe, [], {
            cwd: process.resourcesPath, // Run from resources path so it finds app/ and .pth files
            shell: false
        });
    } else {
        // In development, spawn the uvicorn process using the local python environment
        pythonProcess = spawn('python', ['-m', 'uvicorn', 'app.main:app', '--port', '8000'], {
            cwd: __dirname,
            shell: true
        });
    }

    pythonProcess.stdout.on('data', (data) => {
        console.log(`[Backend]: ${data.toString().trim()}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`[Backend Error]: ${data.toString().trim()}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python backend exited with code ${code}`);
    });

    try {
        console.log('Waiting for backend to initialize models...');
        // Pinging the root endpoint
        await waitForServer('http://127.0.0.1:8000/');
        console.log('Backend ready! Opening application window.');
        createWindow();
    } catch (err) {
        console.error('Failed to connect to Python backend:', err);
        app.quit();
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', () => {
    // Gracefully terminate the Python server to avoid zombie processes
    if (pythonProcess) {
        if (process.platform === 'win32') {
            spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
        } else {
            pythonProcess.kill();
        }
    }
});
