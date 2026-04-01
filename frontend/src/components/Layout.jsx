import React from "react";
import { Outlet } from "react-router-dom";
import Navbar from "./Navbar";
import { Radio, Flame } from "lucide-react";

export default function Layout() {
  return (
    <div className="bg-[#0b0c10] text-gray-100 min-h-screen font-sans overflow-x-hidden relative selection:bg-red-500 selection:text-white">
      {/* Fondo Global */}
      <div className="fixed inset-0 pointer-events-none z-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/10 via-[#0b0c10] to-[#0b0c10]" />

      {/* Parallax Laterales */}
      <div className="parallax-satelite fixed top-10 left-4 md:left-12 z-0 opacity-10 pointer-events-none text-cyan-500 flex flex-col items-center">
        <Radio size={56} />
        <div className="h-64 w-[1px] bg-gradient-to-b from-cyan-500 to-transparent mt-4" />
      </div>
      <div className="parallax-magma fixed bottom-10 right-4 md:right-12 z-0 opacity-10 pointer-events-none text-orange-500 flex flex-col items-center">
        <div className="h-64 w-[1px] bg-gradient-to-t from-orange-500 to-transparent mb-4" />
        <Flame size={56} />
      </div>

      <Navbar />

      {/* Content — pt-16 para compensar el navbar fijo */}
      <main className="relative z-10 pt-16">
        <Outlet />
      </main>

      <footer className="w-full text-center py-8 border-t border-white/5 bg-[#0b0c10] relative z-20">
        <p className="text-[10px] font-mono text-gray-600 tracking-[0.2em] mb-2">UNIVERSIDAD DE SAN BUENAVENTURA</p>
        <p className="text-[10px] font-mono text-gray-700 tracking-widest">INGENIERÍA DE SISTEMAS · IA APLICADA · 2025-2026</p>
      </footer>
    </div>
  );
}
