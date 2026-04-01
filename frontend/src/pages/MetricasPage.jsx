import React, { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, Legend,
} from "recharts";

gsap.registerPlugin(ScrollTrigger);

// ─── DATOS REALES del modelo v7 (evaluation_metrics.json) ───
const barChartData = [
  { name: "Accuracy", value: 92.3, color: "#22c55e" },
  { name: "Precision", value: 91.3, color: "#14b8a6" },
  { name: "Recall", value: 93.2, color: "#3b82f6" },
  { name: "F1-Score", value: 92.2, color: "#8b5cf6" },
];

// Curva ROC real (AUC = 0.9737)
const rocData = [
  { pfp: 0, ptp: 0, baseline: 0 },
  { pfp: 0.02, ptp: 0.45, baseline: 0.02 },
  { pfp: 0.05, ptp: 0.72, baseline: 0.05 },
  { pfp: 0.09, ptp: 0.85, baseline: 0.09 },
  { pfp: 0.15, ptp: 0.91, baseline: 0.15 },
  { pfp: 0.25, ptp: 0.95, baseline: 0.25 },
  { pfp: 0.40, ptp: 0.97, baseline: 0.40 },
  { pfp: 0.60, ptp: 0.99, baseline: 0.60 },
  { pfp: 1, ptp: 1, baseline: 1 },
];

// Historial de entrenamiento real (fase 1 + fase 2)
const trainingHistoryData = [
  { epoch: 1, acc: 0.89, val_acc: 0.72, loss: 0.41, val_loss: 0.58 },
  { epoch: 5, acc: 0.82, val_acc: 0.75, loss: 0.48, val_loss: 0.53 },
  { epoch: 10, acc: 0.79, val_acc: 0.77, loss: 0.46, val_loss: 0.50 },
  { epoch: 15, acc: 0.79, val_acc: 0.78, loss: 0.44, val_loss: 0.48 },
  { epoch: 20, acc: 0.80, val_acc: 0.79, loss: 0.43, val_loss: 0.47 },
  { epoch: 27, acc: 0.80, val_acc: 0.79, loss: 0.42, val_loss: 0.46 },
  // Phase 2 (fine-tuning, LR=0.0001 backbone unfrozen)
  { epoch: 30, acc: 0.83, val_acc: 0.82, loss: 0.37, val_loss: 0.41 },
  { epoch: 40, acc: 0.87, val_acc: 0.87, loss: 0.30, val_loss: 0.33 },
  { epoch: 50, acc: 0.89, val_acc: 0.89, loss: 0.26, val_loss: 0.28 },
  { epoch: 60, acc: 0.91, val_acc: 0.91, loss: 0.22, val_loss: 0.24 },
  { epoch: 70, acc: 0.92, val_acc: 0.92, loss: 0.19, val_loss: 0.21 },
  { epoch: 80, acc: 0.93, val_acc: 0.92, loss: 0.17, val_loss: 0.20 },
];

export default function MetricasPage() {
  const statsRef = useRef(null);
  const chartsRef = useRef(null);
  const histRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      ScrollTrigger.defaults({ toggleActions: "play reverse play reverse" });

      gsap.from(".card-stat", {
        scrollTrigger: { trigger: statsRef.current, start: "top 85%" },
        y: 60, opacity: 0, stagger: 0.12, duration: 0.8, ease: "back.out(1.5)",
      });
      gsap.from(".chart-box", {
        scrollTrigger: { trigger: chartsRef.current, start: "top 80%" },
        scale: 0.92, opacity: 0, duration: 0.8, ease: "power2.out", stagger: 0.2,
      });
      gsap.from(".history-box", {
        scrollTrigger: { trigger: histRef.current, start: "top 80%" },
        y: 50, opacity: 0, duration: 0.8, ease: "power2.out", stagger: 0.2,
      });
    });
    return () => ctx.revert();
  }, []);

  return (
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 w-full">
      {/* Header */}
      <div className="text-center pt-12 mb-16">
        <h2 className="text-4xl md:text-5xl font-black text-white mb-4">Evaluación del Modelo</h2>
        <p className="text-gray-500 font-light text-lg">
          Métricas finales del modelo <span className="text-white font-medium">EfficientNetB0 v7</span> · fase 2 (fine-tuning) · test set: 3,619 imágenes
        </p>
      </div>

      {/* Stats Cards */}
      <div ref={statsRef} className="grid grid-cols-2 lg:grid-cols-6 gap-4 lg:gap-6 mb-16">
        {[
          { label: "ACCURACY", value: "92.3%", color: "text-green-400" },
          { label: "PRECISION", value: "91.3%", color: "text-teal-400" },
          { label: "RECALL", value: "93.2%", color: "text-blue-400" },
          { label: "F1-SCORE", value: "92.2%", color: "text-purple-400" },
          { label: "AUC-ROC", value: "97.4%", color: "text-red-400", highlight: true },
          { label: "MCC", value: "0.846", color: "text-yellow-400" },
        ].map((m) => (
          <div
            key={m.label}
            className={`card-stat bg-white/5 backdrop-blur rounded-[1.5rem] p-6 border flex flex-col items-center ${
              m.highlight ? "border-red-500/30 shadow-[0_0_30px_rgba(239,68,68,0.15)]" : "border-white/10"
            }`}
          >
            <span className={`text-[10px] font-mono mb-2 tracking-widest ${m.highlight ? "text-red-500/80" : "text-gray-500"}`}>
              {m.label}
            </span>
            <span className={`text-3xl font-black ${m.color}`}>{m.value}</span>
          </div>
        ))}
      </div>

      {/* Confusion Matrix */}
      <div className="mb-16 max-w-lg mx-auto">
        <h4 className="text-xl font-bold text-white mb-6 text-center">Matriz de Confusión</h4>
        <div className="bg-[#12141c] rounded-2xl border border-white/5 p-6">
          <div className="grid grid-cols-3 gap-2 text-center text-sm">
            <div />
            <div className="text-gray-400 font-mono text-xs">Pred: No Geo</div>
            <div className="text-gray-400 font-mono text-xs">Pred: Geo</div>
            <div className="text-gray-400 font-mono text-xs text-right pr-2">Real: No Geo</div>
            <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-4">
              <span className="text-2xl font-black text-green-400">1686</span>
              <p className="text-[10px] text-green-400/70 mt-1">TN</p>
            </div>
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
              <span className="text-2xl font-black text-red-400">158</span>
              <p className="text-[10px] text-red-400/70 mt-1">FP</p>
            </div>
            <div className="text-gray-400 font-mono text-xs text-right pr-2">Real: Geo</div>
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
              <span className="text-2xl font-black text-red-400">121</span>
              <p className="text-[10px] text-red-400/70 mt-1">FN</p>
            </div>
            <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-4">
              <span className="text-2xl font-black text-green-400">1651</span>
              <p className="text-[10px] text-green-400/70 mt-1">TP</p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div ref={chartsRef} className="grid md:grid-cols-2 gap-8 w-full mb-16">
        <div className="chart-box bg-white rounded-[2rem] p-8 shadow-2xl min-h-[400px]">
          <h4 className="text-xl font-bold text-gray-800 mb-8 border-b border-gray-100 pb-4">Comparativa de Métricas</h4>
          <div className="h-[280px] w-full text-black">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barChartData} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: "#6b7280", fontSize: 12 }} dy={10} />
                <YAxis tickFormatter={(val) => `${val}%`} axisLine={false} tickLine={false} tick={{ fill: "#9ca3af", fontSize: 12 }} domain={[0, 100]} />
                <Tooltip cursor={{ fill: "#f3f4f6" }} contentStyle={{ borderRadius: "12px", border: "none", boxShadow: "0 10px 15px -3px rgb(0 0 0 / 0.1)" }} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={60}>
                  {barChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-box bg-white rounded-[2rem] p-8 shadow-2xl min-h-[400px]">
          <h4 className="text-xl font-bold text-gray-800 mb-8 border-b border-gray-100 pb-4">Curva ROC (AUC = 0.974)</h4>
          <div className="h-[280px] w-full text-black">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocData} margin={{ top: 10, right: 30, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="pfp" type="number" domain={[0, 1]} tickCount={6} axisLine={false} tickLine={false} tick={{ fill: "#9ca3af", fontSize: 12 }} dy={10} />
                <YAxis domain={[0, 1]} tickCount={6} axisLine={false} tickLine={false} tick={{ fill: "#9ca3af", fontSize: 12 }} />
                <Tooltip contentStyle={{ borderRadius: "12px", border: "none", boxShadow: "0 10px 15px -3px rgb(0 0 0 / 0.1)" }} />
                <Legend verticalAlign="bottom" height={36} iconType="plainline" />
                <Line type="monotone" dataKey="ptp" name="ROC (AUC = 0.974)" stroke="#ef4444" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                <Line type="linear" dataKey="baseline" name="Aleatorio" stroke="#d1d5db" strokeWidth={2} strokeDasharray="5 5" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Training History Charts */}
      <div ref={histRef} className="grid md:grid-cols-2 gap-8 w-full mb-16">
        <div className="history-box bg-white rounded-[2rem] p-8 shadow-2xl min-h-[400px]">
          <h4 className="text-xl font-bold text-gray-800 mb-8 border-b border-gray-100 pb-4">Historial — Accuracy (2 Fases)</h4>
          <div className="h-[280px] w-full text-black">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingHistoryData} margin={{ top: 10, right: 30, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="epoch" type="number" domain={[1, 80]} tickCount={6} axisLine={false} tickLine={false} tick={{ fill: "#9ca3af", fontSize: 12 }} dy={10} />
                <YAxis domain={[0.6, 1]} tickCount={5} axisLine={false} tickLine={false} tick={{ fill: "#9ca3af", fontSize: 12 }} />
                <Tooltip contentStyle={{ borderRadius: "12px", border: "none", boxShadow: "0 10px 15px -3px rgb(0 0 0 / 0.1)" }} />
                <Legend verticalAlign="bottom" height={36} iconType="plainline" />
                <Line type="monotone" dataKey="acc" name="Train Acc" stroke="#22c55e" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="val_acc" name="Val Acc" stroke="#8b5cf6" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="history-box bg-white rounded-[2rem] p-8 shadow-2xl min-h-[400px]">
          <h4 className="text-xl font-bold text-gray-800 mb-8 border-b border-gray-100 pb-4">Historial — Loss (2 Fases)</h4>
          <div className="h-[280px] w-full text-black">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingHistoryData} margin={{ top: 10, right: 30, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="epoch" type="number" domain={[1, 80]} tickCount={6} axisLine={false} tickLine={false} tick={{ fill: "#9ca3af", fontSize: 12 }} dy={10} />
                <YAxis domain={[0, 0.7]} axisLine={false} tickLine={false} tick={{ fill: "#9ca3af", fontSize: 12 }} />
                <Tooltip contentStyle={{ borderRadius: "12px", border: "none", boxShadow: "0 10px 15px -3px rgb(0 0 0 / 0.1)" }} />
                <Legend verticalAlign="bottom" height={36} iconType="plainline" />
                <Line type="monotone" dataKey="loss" name="Train Loss" stroke="#ef4444" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#3b82f6" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Training info */}
      <div className="mb-16 max-w-3xl mx-auto bg-[#12141c]/50 rounded-2xl border border-white/5 p-6">
        <h4 className="text-sm font-mono text-gray-400 tracking-widest mb-4">DATOS DE ENTRENAMIENTO</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center text-sm">
          {[
            { label: "Total imágenes", value: "22,209" },
            { label: "Train", value: "15,037" },
            { label: "Validación", value: "3,553" },
            { label: "Test", value: "3,619" },
            { label: "Zonas únicas", value: "407" },
            { label: "Data leakage", value: "0%" },
            { label: "Split method", value: "GroupShuffle" },
            { label: "Augmentations", value: "10x por img" },
          ].map((d) => (
            <div key={d.label} className="bg-black/20 rounded-xl p-3">
              <p className="text-[10px] text-gray-500 font-mono">{d.label}</p>
              <p className="text-white font-bold">{d.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
