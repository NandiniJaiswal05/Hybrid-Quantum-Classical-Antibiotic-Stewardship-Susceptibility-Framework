import React from 'react';

export default function Navigation() {
  const navItems = ['Home', 'Problem', 'Overview', 'Architecture', 'Methodology', 'Challenges', 'Team', 'Access'];
  return (
    <nav className="fixed w-full bg-slate-900/90 backdrop-blur-md z-50 border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="font-bold text-quantum text-xl tracking-wider">AMR-QGNN</div>
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              {navItems.map(item => (
                <a key={item} href={`#${item.toLowerCase()}`} className="hover:text-quantum px-3 py-2 rounded-md text-sm font-medium transition-colors">
                  {item}
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}