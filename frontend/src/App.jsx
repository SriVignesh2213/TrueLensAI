import React, { useState, useEffect } from 'react';
import { Shield, Scan, Activity, Zap, Github, ExternalLink } from 'lucide-react';
import ImageUploader from './components/ImageUploader';
import RiskMeter from './components/RiskMeter';
import AnalysisResults from './components/AnalysisResults';
import AnalysisHistory from './components/AnalysisHistory';

function Header() {
    return (
        <header className="border-b border-white/5 backdrop-blur-xl bg-surface-900/80 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg shadow-primary-500/20">
                        <Shield className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
                            TrueLens<span className="text-primary-400">AI</span>
                        </h1>
                        <p className="text-[10px] text-white/30 tracking-[0.2em] uppercase">Digital Media Forensics</p>
                    </div>
                </div>
                <div className="flex items-center gap-6">
                    <div className="hidden md:flex items-center gap-4 text-xs text-white/40">
                        <span className="flex items-center gap-1"><Scan className="w-3 h-3" /> CNN</span>
                        <span className="flex items-center gap-1"><Activity className="w-3 h-3" /> FFT</span>
                        <span className="flex items-center gap-1"><Zap className="w-3 h-3" /> ELA</span>
                    </div>
                    <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-white/30 hover:text-white/60 transition-colors">
                        <Github className="w-5 h-5" />
                    </a>
                </div>
            </div>
        </header>
    );
}

function StatsBar({ result }) {
    const stats = [
        { label: 'AI Detection', value: result ? `${(result.ai_probability * 100).toFixed(0)}%` : '—', icon: Scan },
        { label: 'Manipulation', value: result ? `${(result.manipulation_risk * 100).toFixed(0)}%` : '—', icon: Activity },
        { label: 'Confidence', value: result ? `${(result.confidence * 100).toFixed(0)}%` : '—', icon: Zap },
        { label: 'Regions', value: result ? result.suspicious_regions : '—', icon: Shield },
    ];
    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {stats.map((s) => (
                <div key={s.label} className="glass-card p-4 flex items-center gap-3">
                    <s.icon className="w-5 h-5 text-primary-400/60" />
                    <div>
                        <p className="text-xs text-white/40">{s.label}</p>
                        <p className="text-lg font-bold text-white/90">{s.value}</p>
                    </div>
                </div>
            ))}
        </div>
    );
}

export default function App() {
    const [result, setResult] = useState(null);
    const [history, setHistory] = useState([]);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    const handleAnalysisComplete = (data) => {
        setResult(data);
        setHistory((prev) => [{
            analysis_id: data.analysis_id,
            timestamp: data.timestamp,
            filename: data.filename || 'image',
            fraud_risk_score: data.fraud_risk_score,
            fraud_risk_value: data.fraud_risk_value,
            ai_probability: data.ai_probability,
        }, ...prev].slice(0, 50));
    };

    return (
        <div className="min-h-screen bg-surface-900">
            {/* Background gradient */}
            <div className="fixed inset-0 pointer-events-none">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary-600/5 rounded-full blur-3xl" />
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary-800/5 rounded-full blur-3xl" />
            </div>

            <Header />

            <main className="relative max-w-7xl mx-auto px-6 py-8 space-y-8">
                {/* Hero section */}
                {!result && !isAnalyzing && (
                    <div className="text-center py-8 animate-fade-in">
                        <h2 className="text-4xl md:text-5xl font-bold bg-gradient-to-b from-white to-white/40 bg-clip-text text-transparent">
                            Forensic Image Intelligence
                        </h2>
                        <p className="text-white/40 mt-3 max-w-2xl mx-auto text-lg">
                            Multi-layer AI detection combining CNN classification, FFT spectral analysis,
                            EXIF metadata forensics, and Error Level Analysis.
                        </p>
                    </div>
                )}

                {/* Upload */}
                <ImageUploader
                    onAnalysisComplete={handleAnalysisComplete}
                    isAnalyzing={isAnalyzing}
                    setIsAnalyzing={setIsAnalyzing}
                />

                {/* Results */}
                {result && (
                    <div className="space-y-8 animate-fade-in">
                        <StatsBar result={result} />

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            {/* Risk Meter */}
                            <div className="glass-card p-6 flex flex-col items-center justify-center">
                                <h3 className="text-lg font-semibold text-white/90 mb-4">Fraud Risk Score</h3>
                                <RiskMeter value={result.fraud_risk_value} label={result.fraud_risk_score} />
                            </div>

                            {/* Detailed breakdown */}
                            <div className="lg:col-span-2">
                                <AnalysisResults result={result} />
                            </div>
                        </div>
                    </div>
                )}

                {/* History */}
                <AnalysisHistory history={history} />

                {/* Footer */}
                <footer className="text-center py-8 border-t border-white/5">
                    <p className="text-xs text-white/20">
                        TrueLens AI v1.0.0 · Multi-Layer Digital Media Forensics Platform
                    </p>
                    <p className="text-xs text-white/10 mt-1">
                        For research and educational purposes. Not a substitute for professional forensic investigation.
                    </p>
                </footer>
            </main>
        </div>
    );
}
