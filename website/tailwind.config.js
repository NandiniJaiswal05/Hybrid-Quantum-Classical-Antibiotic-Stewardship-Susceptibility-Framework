/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        quantum: '#4F46E5', 
        medical: '#059669', 
        darkbg: '#0F172A',  
      }
    },
  },
  plugins: [],
}
