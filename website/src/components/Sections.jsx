import React from 'react';
import { PROJECT_DATA } from '../config';

const SectionWrapper = ({ id, title, children, dark = false }) => (
  <section id={id} className={`py-20 ${dark ? 'bg-slate-900' : 'bg-darkbg'}`}>
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {title && <h2 className="text-3xl font-bold mb-10 text-white border-b-2 border-quantum inline-block pb-2">{title}</h2>}
      {children}
    </div>
  </section>
);

export const Hero = () => (
  <section id="home" className="pt-32 pb-20 px-4 text-center bg-gradient-to-b from-slate-900 to-darkbg min-h-[90vh] flex flex-col justify-center items-center">
    <div className="inline-block px-4 py-1 rounded-full border border-quantum/30 bg-quantum/10 text-quantum mb-6 text-sm font-semibold tracking-wide">
      Healthcare + AI/ML + Quantum Computing
    </div>
    <h1 className="text-4xl md:text-6xl font-extrabold text-white max-w-4xl leading-tight mb-6">
      {PROJECT_DATA.title}
    </h1>
    <p className="text-xl text-slate-400 max-w-2xl mb-10">{PROJECT_DATA.subtitle}</p>
    <div className="flex gap-4">
      <a href="#architecture" className="bg-quantum hover:bg-indigo-500 text-white px-6 py-3 rounded-lg font-medium transition-all shadow-[0_0_15px_rgba(79,70,229,0.5)]">View Architecture</a>
      <a href="#team" className="border border-slate-600 hover:border-slate-400 text-slate-300 px-6 py-3 rounded-lg font-medium transition-all">Meet the Team</a>
    </div>
  </section>
);

export const ProblemStatement = () => (
  <SectionWrapper id="problem" title="The Challenge" dark>
    <div className="grid md:grid-cols-4 gap-4 text-center">
      {['Antimicrobial Resistance (AMR)', 'Complex Clinical & Molecular Data', 'Difficult Susceptibility Prediction', 'Need for Personalized Stewardship'].map((step, idx) => (
        <div key={idx} className="bg-slate-800 p-6 rounded-xl border border-slate-700 relative">
          <p className="font-semibold text-slate-200">{step}</p>
          {idx < 3 && <div className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 text-quantum text-2xl">→</div>}
          {idx < 3 && <div className="md:hidden text-quantum text-2xl my-2">↓</div>}
        </div>
      ))}
    </div>
    <div className="mt-8 bg-medical/10 border border-medical/30 p-6 rounded-xl text-center">
      <h3 className="text-xl font-bold text-medical mb-2">Proposed Solution</h3>
      <p className="text-slate-300">A Hybrid Classical-Quantum Framework mapping complex features into high-dimensional Hilbert space to capture multi-drug dependencies classical models struggle to represent efficiently.</p>
    </div>
  </SectionWrapper>
);

export const ProjectOverview = () => (
  <SectionWrapper id="overview" title="Project Overview">
    <div className="grid md:grid-cols-2 gap-8">
      <div className="bg-slate-800/50 p-8 rounded-2xl border border-slate-700">
        <h3 className="text-2xl font-bold text-white mb-4">Objective</h3>
        <p className="text-slate-400 leading-relaxed">
          Develop a Hybrid Quantum-Classical Graph Neural Network (QGNN) integrating Electronic Health Records (EHR) and quantum-encoded molecular features to predict antibiotic susceptibility, enabling personalized antibiotic stewardship.
        </p>
      </div>
      <div className="bg-slate-800/50 p-8 rounded-2xl border border-slate-700">
        <h3 className="text-2xl font-bold text-white mb-4">Core Benefits</h3>
        <ul className="space-y-3 text-slate-400">
          <li className="flex items-center gap-2"><span className="text-quantum">✔</span> Personalized antibiotic therapy</li>
          <li className="flex items-center gap-2"><span className="text-quantum">✔</span> Reduces broad-spectrum antibiotic misuse</li>
          <li className="flex items-center gap-2"><span className="text-quantum">✔</span> Combats Antimicrobial Resistance (AMR)</li>
          <li className="flex items-center gap-2"><span className="text-quantum">✔</span> Explainable quantum pharmacology</li>
        </ul>
      </div>
    </div>
  </SectionWrapper>
);

export const SystemArchitecture = () => (
  <SectionWrapper id="architecture" title="System Architecture" dark>
    <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800 overflow-x-auto">
      <div className="min-w-[800px] flex flex-col gap-6 text-sm">
        <div className="flex justify-center"><div className="bg-blue-900/50 border border-blue-500 px-6 py-3 rounded-lg text-blue-200">User / Client Layer: Clinician / Physician</div></div>
        <div className="text-center text-slate-600">↓</div>
        <div className="flex justify-center"><div className="bg-emerald-900/50 border border-emerald-500 px-6 py-3 rounded-lg text-emerald-200 w-1/2 text-center">Frontend Layer: Streamlit UI (Patient Intake & Drug History)</div></div>
        <div className="text-center text-slate-600">↓</div>
        <div className="flex justify-center"><div className="bg-orange-900/50 border border-orange-500 p-4 rounded-lg text-orange-200 w-2/3 text-center">Backend Layer (ETL): Request Assembler → KNN Imputation → StandardScaler (Z-Score)</div></div>
        <div className="text-center text-slate-600">↓</div>
        <div className="bg-purple-900/20 border border-purple-500/50 p-6 rounded-xl relative">
          <h4 className="absolute -top-3 left-4 bg-slate-900 px-2 text-purple-400 font-bold">AI/ML Processing Layer</h4>
          <div className="flex flex-col items-center gap-4">
            <div className="bg-purple-900/50 border border-purple-500 px-6 py-2 rounded text-purple-200">Kernel PCA (8 Dimensions)</div>
            <div className="text-slate-600">↓</div>
            <div className="bg-purple-900/50 border border-purple-500 px-6 py-2 rounded text-purple-200">PennyLane Angle Embedding</div>
            <div className="text-slate-600">↓</div>
            <div className="bg-rose-900/50 border border-rose-500 px-6 py-4 rounded-full text-rose-200 font-bold">Gradient Variance Check (Barren Plateau Detection)</div>
            
            <div className="grid grid-cols-3 gap-4 w-full mt-4">
              <div className="text-center border border-slate-700 p-3 rounded bg-slate-800">
                <span className="text-xs text-slate-400 block mb-1">Healthy Gradient</span>
                <strong className="text-quantum">Path A: Variational QNN</strong>
                <div className="text-xs mt-2 text-blue-400">↔ AWS Braket SV1 / QPU</div>
              </div>
              <div className="text-center border border-slate-700 p-3 rounded bg-slate-800">
                <span className="text-xs text-slate-400 block mb-1">Flat Landscape</span>
                <strong className="text-quantum">Path B: Quantum SVM</strong>
              </div>
              <div className="text-center border border-slate-700 p-3 rounded bg-slate-800">
                <span className="text-xs text-slate-400 block mb-1">Timeout / Error</span>
                <strong className="text-quantum">Path C: Penalized XGBoost</strong>
              </div>
            </div>
          </div>
        </div>
        <div className="text-center text-slate-600">↓</div>
        <div className="bg-red-900/20 border border-red-500/50 p-6 rounded-xl flex flex-col items-center gap-3">
          <div className="bg-red-900/50 border border-red-500 px-6 py-2 rounded text-red-200">Susceptibility Score Calculation</div>
          <div className="bg-red-900/50 border border-red-500 px-6 py-2 rounded text-red-200">Uncertainty Budget (Entropy)</div>
          <div className="bg-red-900/50 border border-red-500 px-6 py-2 rounded text-red-200 font-bold">Clinical Alert Decision (Tuned Threshold)</div>
        </div>
      </div>
    </div>
  </SectionWrapper>
);

export const Methodology = () => (
  <SectionWrapper id="methodology" title="Methodology">
    <div className="space-y-4">
      {[
        { step: 1, title: 'Data Ingestion', desc: 'Clinical & Molecular Data collection.' },
        { step: 2, title: 'ETL Processing', desc: 'KNN Imputation and StandardScaler for robust baseline.' },
        { step: 3, title: 'Dimensionality Reduction', desc: 'Kernel PCA compresses features to 8 Dimensions.' },
        { step: 4, title: 'Quantum Encoding', desc: 'PennyLane Angle Embedding translates classical data to quantum states.' },
        { step: 5, title: 'Dynamic Routing', desc: 'Gradient Variance Check triggers Path A (VQNN), B (QSVM), or C (XGBoost).' },
        { step: 6, title: 'Clinical Decision', desc: 'Entropy-Based Uncertainty and Susceptibility Score evaluate the alert threshold.' }
      ].map(m => (
        <div key={m.step} className="flex items-start gap-4 p-4 bg-slate-800/30 rounded-lg border border-slate-700/50">
          <div className="bg-quantum text-white w-8 h-8 rounded-full flex items-center justify-center font-bold shrink-0">{m.step}</div>
          <div>
            <h4 className="text-lg font-bold text-slate-200">{m.title}</h4>
            <p className="text-slate-400 text-sm">{m.desc}</p>
          </div>
        </div>
      ))}
    </div>
  </SectionWrapper>
);

export const Challenges = () => (
  <SectionWrapper id="challenges" title="Challenges & Mitigations" dark>
    <div className="grid md:grid-cols-2 gap-6">
      <div className="border border-slate-700 p-5 rounded-xl bg-slate-800">
        <strong className="text-red-400 block mb-2">Challenge: Dataset Imbalance & Low Recall</strong>
        <p className="text-slate-300 text-sm mb-2">Initial model heavily favored susceptible cases yielding 31% recall.</p>
        <div className="bg-emerald-900/30 border border-emerald-500/30 p-3 rounded text-sm text-emerald-200">
          <strong>Solution:</strong> SMOTE oversampling and optimizing clinical threshold to 0.5948 (doubled recall to 60%).
        </div>
      </div>
      <div className="border border-slate-700 p-5 rounded-xl bg-slate-800">
        <strong className="text-red-400 block mb-2">Challenge: Barren Plateaus & Quantum Noise</strong>
        <p className="text-slate-300 text-sm mb-2">Variational circuits faced gradient vanishing and QPU timeouts.</p>
        <div className="bg-emerald-900/30 border border-emerald-500/30 p-3 rounded text-sm text-emerald-200">
          <strong>Solution:</strong> Gradient variance monitoring and Tri-Stage Cascading Fallback (VQNN → QSVM → XGBoost).
        </div>
      </div>
    </div>
  </SectionWrapper>
);

export const Results = () => (
  <SectionWrapper id="results" title="Preliminary Results">
    <div className="bg-slate-800 p-8 rounded-2xl text-center border border-slate-700">
      <div className="grid md:grid-cols-3 gap-6 mb-6">
        <div className="p-4 bg-slate-900 rounded-lg">
          <div className="text-3xl font-black text-slate-400 mb-1">{PROJECT_DATA.results.baselineRecall}</div>
          <div className="text-sm text-slate-500 uppercase tracking-wide">Baseline Recall</div>
        </div>
        <div className="p-4 bg-quantum/20 border border-quantum/50 rounded-lg shadow-[0_0_15px_rgba(79,70,229,0.2)]">
          <div className="text-3xl font-black text-white mb-1">{PROJECT_DATA.results.improvedRecall}</div>
          <div className="text-sm text-quantum uppercase tracking-wide">Optimized Recall</div>
        </div>
        <div className="p-4 bg-slate-900 rounded-lg">
          <div className="text-3xl font-black text-medical mb-1">{PROJECT_DATA.results.optimalThreshold}</div>
          <div className="text-sm text-slate-500 uppercase tracking-wide">Tuned Threshold</div>
        </div>
      </div>
      <p className="text-slate-400 text-sm">{PROJECT_DATA.results.status}</p>
    </div>
  </SectionWrapper>
);

export const TechStack = () => (
  <SectionWrapper id="tech" title="Technology Stack" dark>
    <div className="flex flex-wrap justify-center gap-4">
      {['Python', 'Scikit-Learn', 'XGBoost', 'PennyLane', 'AWS Braket (SV1 Simulator)', 'Streamlit'].map(tech => (
        <span key={tech} className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-full text-slate-300 font-medium">
          {tech}
        </span>
      ))}
    </div>
  </SectionWrapper>
);

export const Team = () => (
  <SectionWrapper id="team" title="Project Team">
    <div className="mb-8 text-center">
      <p className="text-lg text-slate-300"><strong>Guide:</strong> {PROJECT_DATA.guide}</p>
      <p className="text-slate-400">{PROJECT_DATA.department} | {PROJECT_DATA.college} | Group {PROJECT_DATA.group}</p>
    </div>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
      {PROJECT_DATA.team.map((member, idx) => (
        <div key={idx} className="bg-slate-800 border border-slate-700 p-6 rounded-xl text-center">
          <div className="w-16 h-16 bg-slate-700 rounded-full mx-auto mb-4 flex items-center justify-center text-xl font-bold text-slate-500">
            {member.name.charAt(0)}
          </div>
          <h4 className="font-bold text-slate-200">{member.name}</h4>
          <p className="text-sm text-quantum mt-1">{member.role}</p>
        </div>
      ))}
    </div>
  </SectionWrapper>
);

export const Contact = () => (
  <SectionWrapper id="access" title="Access & Contact" dark>
    <div className="max-w-2xl mx-auto text-center space-y-6">
      <div className="flex flex-col sm:flex-row justify-center gap-4">
        <a href={PROJECT_DATA.links.demo} className="bg-quantum hover:bg-indigo-500 text-white px-6 py-3 rounded-lg font-medium transition-all">View Live Demo</a>
        <a href={PROJECT_DATA.links.github} className="border border-slate-600 hover:border-slate-400 text-slate-300 px-6 py-3 rounded-lg font-medium transition-all">Source Code (GitHub)</a>
      </div>
      <p className="text-slate-400">Contact: <a href={`mailto:${PROJECT_DATA.links.email}`} className="text-quantum hover:underline">{PROJECT_DATA.links.email}</a></p>
      <div className="mt-12 p-4 bg-red-900/10 border border-red-900/30 rounded-lg">
        <p className="text-xs text-slate-500 text-left">{PROJECT_DATA.disclaimer}</p>
      </div>
    </div>
  </SectionWrapper>
);