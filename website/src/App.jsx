import React from 'react';
import { Hero, ProblemStatement, ProjectOverview, SystemArchitecture, Methodology, Challenges, Results, TechStack, Team, Contact } from './components/Sections';
import Navigation from './components/Navigation';

export default function App() {
  return (
    <div className="bg-darkbg text-slate-200 min-h-screen font-sans">
      <Navigation />
      <main>
        <Hero />
        <ProblemStatement />
        <ProjectOverview />
        <SystemArchitecture />
        <Methodology />
        <Challenges />
        <Results />
        <TechStack />
        <Team />
        <Contact />
      </main>
      <footer className="bg-slate-950 py-6 text-center text-sm text-slate-500 border-t border-slate-800">
        <p>© 2026 GHRCE AI Dept - Group 13. All rights reserved.</p>
      </footer>
    </div>
  );
}