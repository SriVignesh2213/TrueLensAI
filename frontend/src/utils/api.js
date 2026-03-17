const API_BASE = '/api/v1';

export async function analyzeImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/analyze-image`, { method: 'POST', body: formData });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

export async function getAnalysis(id) {
    const res = await fetch(`${API_BASE}/analysis/${id}`);
    if (!res.ok) throw new Error(`Analysis ${id} not found`);
    return res.json();
}

export async function getHistory(limit = 50) {
    const res = await fetch(`${API_BASE}/history?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch history');
    return res.json();
}

export async function healthCheck() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
}
