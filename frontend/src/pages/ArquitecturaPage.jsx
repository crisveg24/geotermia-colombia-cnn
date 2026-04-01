import React, { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Database, Activity, Cpu, Layers } from "lucide-react";

gsap.registerPlugin(ScrollTrigger);

export default function ArquitecturaPage() {
  const tableRef = useRef(null);
  const hyperRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from(tableRef.current, {
        y: 60, opacity: 0, duration: 1, ease: "power3.out",
      });
      gsap.from(".hyper-card", {
        scrollTrigger: { trigger: hyperRef.current, start: "top 80%" },
        x: -50, opacity: 0, duration: 0.8, ease: "power3.out", stagger: 0.25,
      });
    });
    return () => ctx.revert();
  }, []);

  return (
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 w-full">
      {/* Header */}
      <div className="text-center pt-12 mb-12">
        <h2 className="text-4xl md:text-5xl font-black text-white mb-4">Arquitectura del Modelo CNN</h2>
        <p className="text-gray-500 font-light text-lg max-w-3xl mx-auto">
          <span className="text-white font-medium">EfficientNetB0</span> con Channel Adapter personalizado (7→16→3 canales),
          transfer learning en 2 fases con mixed precision float16.
        </p>
      </div>

      {/* Training phases */}
      <div className="max-w-4xl mx-auto mb-12 grid md:grid-cols-2 gap-6">
        <div className="bg-blue-500/5 border border-blue-500/20 rounded-2xl p-6">
          <h4 className="text-blue-400 font-bold text-sm mb-4 font-mono tracking-wider">FASE 1 — BACKBONE CONGELADO</h4>
          <ul className="space-y-2 text-sm text-gray-300">
            <li className="flex justify-between"><span>Épocas</span><span className="font-mono text-white">30 (best: 27)</span></li>
            <li className="flex justify-between"><span>Learning Rate</span><span className="font-mono text-blue-400">0.001</span></li>
            <li className="flex justify-between"><span>Parámetros entrenables</span><span className="font-mono text-white">345,863</span></li>
            <li className="flex justify-between"><span>Best Val AUC</span><span className="font-mono text-blue-400">0.900</span></li>
            <li className="flex justify-between"><span>Test Accuracy</span><span className="font-mono text-white">76.4%</span></li>
          </ul>
        </div>
        <div className="bg-orange-500/5 border border-orange-500/20 rounded-2xl p-6">
          <h4 className="text-orange-400 font-bold text-sm mb-4 font-mono tracking-wider">FASE 2 — FINE-TUNING</h4>
          <ul className="space-y-2 text-sm text-gray-300">
            <li className="flex justify-between"><span>Épocas</span><span className="font-mono text-white">50 (best: 50)</span></li>
            <li className="flex justify-between"><span>Learning Rate</span><span className="font-mono text-orange-400">0.0001</span></li>
            <li className="flex justify-between"><span>Capas descongeladas</span><span className="font-mono text-white">39</span></li>
            <li className="flex justify-between"><span>Parámetros entrenables</span><span className="font-mono text-white">2,633,367</span></li>
            <li className="flex justify-between"><span>Best Val AUC</span><span className="font-mono text-orange-400">0.973</span></li>
            <li className="flex justify-between"><span>Test Accuracy</span><span className="font-mono font-bold text-green-400">92.3%</span></li>
          </ul>
        </div>
      </div>

      {/* Architecture Table */}
      <div ref={tableRef} className="max-w-5xl mx-auto bg-[#12141c] backdrop-blur-md rounded-[2rem] border border-white/5 shadow-2xl p-2 mb-16">
        <table className="w-full text-left border-collapse rounded-[1.8rem] overflow-hidden">
          <thead>
            <tr className="bg-[#1a1d27] uppercase text-[10px] tracking-widest text-gray-500">
              <th className="p-6 font-medium">Capa / Componente</th>
              <th className="p-6 font-medium">Detalles</th>
              <th className="p-6 font-medium text-right">Salida Tensor</th>
            </tr>
          </thead>
          <tbody className="text-sm font-mono text-gray-300">
            <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
              <td className="p-5 pl-6 text-cyan-400 font-bold">Input Layer</td>
              <td className="p-5 text-gray-400">7 bandas ASTER</td>
              <td className="p-5 pr-6 text-right">224×224×7</td>
            </tr>
            <tr className="border-b border-white/5 hover:bg-cyan-500/5 transition-colors">
              <td className="p-5 pl-6 text-cyan-400 font-bold border-l-2 border-cyan-500">Channel Adapter</td>
              <td className="p-5"><span className="text-white">Conv2D 1×1</span> <span className="text-gray-500">7→16→3</span></td>
              <td className="p-5 pr-6 text-right font-bold text-white">224×224×3</td>
            </tr>
            <tr className="border-b border-white/5 hover:bg-green-500/5 transition-colors">
              <td className="p-5 pl-6 text-green-400 font-bold border-l-2 border-green-500">EfficientNetB0</td>
              <td className="p-5 text-gray-400">Backbone pre-entrenado ImageNet</td>
              <td className="p-5 pr-6 text-right">7×7×1280</td>
            </tr>
            <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
              <td className="p-5 pl-6 text-yellow-500 font-bold">Global Average Pooling</td>
              <td className="p-5 opacity-30">—</td>
              <td className="p-5 pr-6 text-right text-white">1280</td>
            </tr>
            <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
              <td className="p-5 pl-6"><span className="text-teal-400">Dense 1</span> <span className="text-[10px] text-gray-500">+ BN + Dropout</span></td>
              <td className="p-5 text-gray-400">256 units, no bias</td>
              <td className="p-5 pr-6 text-right font-bold text-white">256</td>
            </tr>
            <tr className="border-t-2 border-red-500/50 hover:bg-red-500/10 transition-colors bg-red-500/5">
              <td className="p-5 pl-6 text-red-500 font-black">Output</td>
              <td className="p-5 text-red-400 text-xs">Sigmoid</td>
              <td className="p-5 pr-6 text-right font-bold text-red-500 text-lg">1</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Hyperparams + Datos */}
      <div ref={hyperRef} className="max-w-6xl mx-auto mb-16 grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="hyper-card bg-[#12141c]/80 backdrop-blur-md rounded-3xl border border-white/5 p-8 shadow-2xl relative overflow-hidden">
          <div className="absolute -right-10 -top-10 text-cyan-500 opacity-5">
            <Database size={150} strokeWidth={1} />
          </div>
          <h3 className="text-2xl font-bold text-white mb-6 border-b border-white/10 pb-4 flex items-center gap-3">
            <Database className="text-cyan-400" /> Datos de Entrada
          </h3>
          <ul className="flex flex-col gap-4 text-sm font-light text-gray-300">
            <li className="flex justify-between"><strong className="text-white">Fuente</strong><span className="font-mono text-xs text-right">NASA ASTER GED (AG100) V003</span></li>
            <li className="flex justify-between"><strong className="text-white">Resolución</strong><span className="font-mono text-xs">90 metros</span></li>
            <li className="flex justify-between"><strong className="text-white">Bandas</strong><span className="font-mono text-xs text-cyan-400">5 TIR + Temp + NDVI (7)</span></li>
            <li className="flex justify-between"><strong className="text-white">Entrada Tensor</strong><span className="font-mono text-xs font-bold text-white">224 × 224 × 7</span></li>
            <li className="flex justify-between"><strong className="text-white">Normalización</strong><span className="font-mono text-xs">Z-score (band_stats_v3)</span></li>
            <li className="flex justify-between"><strong className="text-white">Dataset Total</strong><span className="font-mono text-xs text-green-400">22,209 imágenes</span></li>
          </ul>
        </div>

        <div className="hyper-card bg-[#12141c]/80 backdrop-blur-md rounded-3xl border border-white/5 p-8 shadow-2xl relative overflow-hidden">
          <div className="absolute -right-10 -top-10 text-orange-500 opacity-5">
            <Layers size={150} strokeWidth={1} />
          </div>
          <h3 className="text-2xl font-bold text-white mb-6 border-b border-white/10 pb-4 flex items-center gap-3">
            <Activity className="text-orange-400" /> Hiperparámetros
          </h3>
          <ul className="flex flex-col gap-4 text-sm font-light text-gray-300">
            <li className="flex justify-between"><strong className="text-white">Optimizador</strong><span className="font-mono text-xs text-right">AdamW <span className="opacity-50">(wd=1e-4)</span></span></li>
            <li className="flex justify-between"><strong className="text-white">Learning Rate</strong><span className="font-mono text-xs text-orange-400 font-bold">1e-3 → 1e-4</span></li>
            <li className="flex justify-between gap-4"><strong className="text-white whitespace-nowrap">Loss</strong><span className="font-mono text-[11px] text-right">BinaryCrossentropy <span className="opacity-50">(ls=0.1)</span></span></li>
            <li className="flex justify-between"><strong className="text-white">Mixed Precision</strong><span className="font-mono text-xs text-cyan-400">float16</span></li>
            <li className="flex justify-between"><strong className="text-white">Batch Size</strong><span className="font-mono text-xs text-white">32</span></li>
            <li className="flex justify-between"><strong className="text-white">Épocas</strong><span className="font-mono text-xs">30 + 50 <span className="text-red-400">(2 fases)</span></span></li>
            <li className="flex justify-between pt-2 border-t border-white/5"><strong className="text-white uppercase tracking-wider text-xs">Total Parámetros</strong><span className="font-mono text-sm text-yellow-400 font-bold">4,396,112</span></li>
          </ul>
        </div>
      </div>
    </div>
  );
}
