import React, { useState } from 'react';
import { Eye, EyeOff, AlertTriangle, CheckCircle, Info } from 'lucide-react';

function ScoreBar({ label, value, color }) {
    return (
        <div className="space-y-1.5">
            <div className="flex justify-between text-sm">
                <span className="text-white/60">{label}</span>
                <span className="font-semibold" style={{ color }}>{(value * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-1000 ease-out" style={{ width: `${value * 100}%`, background: `linear-gradient(90deg, ${color}66, ${color})` }} />
            </div>
        </div>
    );
}

export default function AnalysisResults({ result }) {
    const [showHeatmap, setShowHeatmap] = useState(false);

    if (!result) return null;

    const riskColor = { CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#eab308', LOW: '#22c55e', MINIMAL: '#06b6d4' }[result.fraud_risk_score] || '#06b6d4';

    return (
        <div className="space-y-6 animate-fade-in">
            {/* Score Bars */}
            <div className="glass-card p-6 space-y-4">
                <h3 className="text-lg font-semibold text-white/90 flex items-center gap-2">
                    <Info className="w-5 h-5 text-primary-400" /> Detection Breakdown
                </h3>
                <ScoreBar label="AI Generation Probability" value={result.ai_probability} color="#818cf8" />
                <ScoreBar label="Manipulation Risk" value={result.manipulation_risk} color="#f97316" />
                <ScoreBar label="Metadata Anomaly Score" value={result.metadata_anomaly_score} color="#eab308" />
                <ScoreBar label="Frequency Anomaly Score" value={result.frequency_anomaly_score} color="#06b6d4" />
            </div>

            {/* Metadata Anomaly Badge */}
            <div className="glass-card p-4 flex items-center gap-3">
                {result.metadata_anomaly ? (
                    <><AlertTriangle className="w-6 h-6 text-yellow-400" /><span className="text-yellow-300 font-medium">Metadata Anomalies Detected</span></>
                ) : (
                    <><CheckCircle className="w-6 h-6 text-green-400" /><span className="text-green-300 font-medium">Metadata Appears Normal</span></>
                )}
            </div>

            {/* Heatmap Toggle */}
            {result.heatmap_available && result.heatmap_base64 && (
                <div className="glass-card p-4">
                    <button onClick={() => setShowHeatmap(!showHeatmap)} className="flex items-center gap-2 text-sm font-medium text-primary-300 hover:text-primary-200 transition-colors">
                        {showHeatmap ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        {showHeatmap ? 'Hide' : 'Show'} Forensic Heatmap
                    </button>
                    {showHeatmap && (
                        <img src={`data:image/png;base64,${result.heatmap_base64}`} alt="Forensic heatmap" className="mt-4 rounded-xl border border-white/10 w-full max-w-sm mx-auto" />
                    )}
                </div>
            )}

            {/* Recommendations */}
            {result.recommendations?.length > 0 && (
                <div className="glass-card p-6">
                    <h3 className="text-lg font-semibold text-white/90 mb-3">Recommendations</h3>
                    <div className="space-y-2">
                        {result.recommendations.map((rec, i) => (
                            <div key={i} className="flex gap-3 p-3 rounded-xl bg-white/5 text-sm text-white/70">
                                <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                                <span>{rec}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Analysis ID */}
            <p className="text-xs text-white/20 text-center font-mono">ID: {result.analysis_id}</p>
        </div>
    );
}
