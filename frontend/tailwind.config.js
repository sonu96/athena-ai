/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'athena-primary': '#6366f1',
        'athena-secondary': '#8b5cf6',
        'athena-dark': '#1e1b4b',
        'athena-success': '#10b981',
        'athena-warning': '#f59e0b',
        'athena-error': '#ef4444',
      }
    },
  },
  plugins: [],
}