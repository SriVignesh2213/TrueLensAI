/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
    theme: {
        extend: {
            colors: {
                primary: { 50: '#eef2ff', 100: '#e0e7ff', 200: '#c7d2fe', 300: '#a5b4fc', 400: '#818cf8', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca', 800: '#3730a3', 900: '#312e81', 950: '#1e1b4b' },
                surface: { 50: '#f8fafc', 100: '#f1f5f9', 200: '#e2e8f0', 700: '#1e293b', 800: '#0f172a', 900: '#020617' },
                risk: { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#22c55e', minimal: '#06b6d4' },
            },
            fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
            animation: { 'pulse-slow': 'pulse 3s infinite', 'fade-in': 'fadeIn 0.5s ease-out' },
            keyframes: { fadeIn: { '0%': { opacity: '0', transform: 'translateY(10px)' }, '100%': { opacity: '1', transform: 'translateY(0)' } } },
        },
    },
    plugins: [],
}
