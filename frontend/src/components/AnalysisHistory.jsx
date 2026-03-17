import React from 'react';
import { Clock, AlertTriangle, CheckCircle, ChevronRight } from 'lucide-react';

const RISK_COLORS = {
    CRITICAL: 'text-red-400 bg-red-500/10',
    HIGH: 'text-orange-400 bg-orange-500/10',
    MEDIUM: 'text-yellow-400 bg-yellow-500/10',
    LOW: 'text-green-400 bg-green-500/10',
    MINIMAL: 'text-cyan-400 bg-cyan-500/10',
};

export default function AnalysisHistory({ history, onSelect }) {
    if (!history || history.length === 0) {
        return (
            <div className="glass-card p-8 text-center">
                <Clock className="w-12 h-12 text-white/20 mx-auto mb-3" />
                <p className="text-white/40">No analyses yet. Upload an image to get started.</p>
            </div>
        );
    }

    return (
        <div className="glass-card overflow-hidden">
            <div className="p-4 border-b border-white/10">
                <h3 className="font-semibold text-white/90 flex items-center gap-2">
                    <Clock className="w-5 h-5 text-primary-400" /> Analysis History
                </h3>
            </div>
            <div className="divide-y divide-white/5 max-h-96 overflow-y-auto">
                {history.map((item) => (
                    <button key={item.analysis_id} onClick={() => onSelect?.(item.analysis_id)}
                        className="w-full flex items-center gap-4 p-4 hover:bg-white/5 transition-colors text-left group">
                        <div className="flex-shrink-0">
                            {item.fraud_risk_value > 0.5 ? (
                                <AlertTriangle className="w-5 h-5 text-orange-400" />
                            ) : (
                                <CheckCircle className="w-5 h-5 text-green-400" />
                            )}
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-white/80 truncate">{item.filename}</p>
                            <p className="text-xs text-white/30 mt-0.5">{new Date(item.timestamp).toLocaleString()}</p>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className={`risk-badge ${RISK_COLORS[item.fraud_risk_score] || 'text-gray-400 bg-gray-500/10'}`}>
                                {item.fraud_risk_score}
                            </span>
                            <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-white/40 transition-colors" />
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
}
