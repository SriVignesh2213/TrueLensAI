import React, { useCallback, useState } from 'react';
import { Upload, Shield, AlertTriangle } from 'lucide-react';

export default function ImageUploader({ onAnalysisComplete, isAnalyzing, setIsAnalyzing }) {
    const [dragActive, setDragActive] = useState(false);
    const [preview, setPreview] = useState(null);
    const [error, setError] = useState(null);

    const handleFile = useCallback(async (file) => {
        if (!file) return;
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            setError('Unsupported file type. Use JPG, PNG, WebP, BMP, or TIFF.');
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            setError('File too large. Maximum 20MB.');
            return;
        }
        setError(null);
        setPreview(URL.createObjectURL(file));
        setIsAnalyzing(true);

        try {
            const { analyzeImage } = await import('../utils/api.js');
            const result = await analyzeImage(file);
            onAnalysisComplete(result);
        } catch (err) {
            setError(err.message || 'Analysis failed');
        } finally {
            setIsAnalyzing(false);
        }
    }, [onAnalysisComplete, setIsAnalyzing]);

    const handleDrop = (e) => { e.preventDefault(); setDragActive(false); handleFile(e.dataTransfer.files[0]); };
    const handleDragOver = (e) => { e.preventDefault(); setDragActive(true); };
    const handleDragLeave = () => setDragActive(false);

    return (
        <div className="w-full">
            <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer group
          ${dragActive
                        ? 'border-primary-400 bg-primary-500/10 scale-[1.02]'
                        : 'border-white/20 hover:border-primary-400/50 hover:bg-white/5'
                    }
          ${isAnalyzing ? 'pointer-events-none opacity-60' : ''}
        `}
            >
                {isAnalyzing ? (
                    <div className="flex flex-col items-center gap-4 py-8">
                        <div className="w-16 h-16 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
                        <p className="text-lg font-medium text-primary-300">Running forensic analysis...</p>
                        <div className="flex gap-2 text-sm text-white/50">
                            <Shield className="w-4 h-4" /> CNN • FFT • Metadata • ELA
                        </div>
                    </div>
                ) : (
                    <label className="flex flex-col items-center gap-4 py-8 cursor-pointer">
                        <div className={`w-20 h-20 rounded-2xl flex items-center justify-center transition-all duration-300
              ${dragActive ? 'bg-primary-500/20' : 'bg-gradient-to-br from-primary-600/20 to-primary-800/20 group-hover:from-primary-500/30 group-hover:to-primary-700/30'}`}>
                            <Upload className={`w-10 h-10 transition-all duration-300 ${dragActive ? 'text-primary-400 scale-110' : 'text-primary-400/60 group-hover:text-primary-400'}`} />
                        </div>
                        <div>
                            <p className="text-lg font-semibold text-white/90">Drop an image or click to upload</p>
                            <p className="text-sm text-white/40 mt-1">JPG, PNG, WebP, BMP, TIFF · Max 20MB</p>
                        </div>
                        <input type="file" className="hidden" accept="image/*" onChange={(e) => handleFile(e.target.files[0])} />
                    </label>
                )}

                {preview && !isAnalyzing && (
                    <div className="mt-4 flex justify-center">
                        <img src={preview} alt="Preview" className="max-h-32 rounded-xl border border-white/10 shadow-lg" />
                    </div>
                )}
            </div>

            {error && (
                <div className="mt-4 flex items-center gap-2 text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                    <span className="text-sm">{error}</span>
                </div>
            )}
        </div>
    );
}
