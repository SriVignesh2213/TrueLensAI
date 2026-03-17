import React from 'react';

const RISK_CONFIG = {
    CRITICAL: { color: '#ef4444', bg: 'rgba(239,68,68,0.1)', label: 'CRITICAL' },
    HIGH: { color: '#f97316', bg: 'rgba(249,115,22,0.1)', label: 'HIGH' },
    MEDIUM: { color: '#eab308', bg: 'rgba(234,179,8,0.1)', label: 'MEDIUM' },
    LOW: { color: '#22c55e', bg: 'rgba(34,197,94,0.1)', label: 'LOW' },
    MINIMAL: { color: '#06b6d4', bg: 'rgba(6,182,212,0.1)', label: 'MINIMAL' },
};

export default function RiskMeter({ value = 0, label = 'MINIMAL' }) {
    const config = RISK_CONFIG[label] || RISK_CONFIG.MINIMAL;
    const angle = value * 180; // 0-180 degrees for semicircle

    return (
        <div className="flex flex-col items-center">
            <div className="relative w-56 h-28 overflow-hidden">
                {/* Background arc */}
                <svg viewBox="0 0 200 100" className="w-full h-full">
                    <defs>
                        <linearGradient id="meterGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#06b6d4" />
                            <stop offset="25%" stopColor="#22c55e" />
                            <stop offset="50%" stopColor="#eab308" />
                            <stop offset="75%" stopColor="#f97316" />
                            <stop offset="100%" stopColor="#ef4444" />
                        </linearGradient>
                    </defs>
                    <path d="M 15 95 A 85 85 0 0 1 185 95" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="18" strokeLinecap="round" />
                    <path d="M 15 95 A 85 85 0 0 1 185 95" fill="none" stroke="url(#meterGrad)" strokeWidth="18" strokeLinecap="round"
                        strokeDasharray={`${value * 267} 267`}
                        style={{ transition: 'stroke-dasharray 1.5s cubic-bezier(0.4,0,0.2,1)' }} />
                    {/* Needle */}
                    <line
                        x1="100" y1="95"
                        x2={100 + 70 * Math.cos(Math.PI - (angle * Math.PI / 180))}
                        y2={95 - 70 * Math.sin(Math.PI - (angle * Math.PI / 180))}
                        stroke={config.color}
                        strokeWidth="3"
                        strokeLinecap="round"
                        style={{ transition: 'all 1.5s cubic-bezier(0.4,0,0.2,1)' }}
                    />
                    <circle cx="100" cy="95" r="6" fill={config.color} style={{ transition: 'fill 0.5s' }} />
                </svg>
            </div>
            <div className="mt-2 text-center">
                <span className="text-3xl font-bold" style={{ color: config.color, transition: 'color 0.5s' }}>
                    {(value * 100).toFixed(0)}%
                </span>
                <div className="mt-1">
                    <span className="risk-badge" style={{ background: config.bg, color: config.color, border: `1px solid ${config.color}33` }}>
                        {config.label} RISK
                    </span>
                </div>
            </div>
        </div>
    );
}
