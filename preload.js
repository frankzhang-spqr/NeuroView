const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // File menu
    onOpenFiles: (callback) => ipcRenderer.on('menu-open-files', (_event, filePaths) => callback(filePaths)),
    onResetWorkspace: (callback) => ipcRenderer.on('menu-reset', () => callback()),
    onExportResults: (callback) => ipcRenderer.on('menu-export', () => callback()),

    // Edit menu
    onCopySlice: (callback) => ipcRenderer.on('menu-copy-slice', () => callback()),
    onPreferences: (callback) => ipcRenderer.on('menu-preferences', () => callback()),

    // View menu
    onToggleSidebar: (callback) => ipcRenderer.on('menu-toggle-sidebar', () => callback()),
    onToggleStatusBar: (callback) => ipcRenderer.on('menu-toggle-statusbar', () => callback()),

    // Tools menu
    onRunAnalysis: (callback) => ipcRenderer.on('menu-run-analysis', () => callback()),
    onChangeAxis: (callback) => ipcRenderer.on('menu-change-axis', (_event, axis) => callback(axis)),
    onToggleOverlay: (callback) => ipcRenderer.on('menu-toggle-overlay', () => callback()),

    // Help menu
    onOpenDocs: (callback) => ipcRenderer.on('menu-docs', () => callback()),
    onShowShortcuts: (callback) => ipcRenderer.on('menu-shortcuts', () => callback())
});
