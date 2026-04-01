import React, { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { gsap } from "gsap";
import { Crosshair, BarChart3, Cpu, Layers } from "lucide-react";

export default function HomePage() {
  const heroRef = useRef(null);
  const cardsRef = useRef(null);
  const statsRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Asegurar visibilidad antes de animar
      gsap.set(heroRef.current, { opacity: 1 });
      gsap.set(".hero-card", { opacity: 1 });
      gsap.set(".cta-btn", { opacity: 1 });

      gsap.from(heroRef.current, { y: 50, opacity: 0, duration: 1.2, ease: "power4.out" });
      gsap.from(".hero-card", {
        y: 30,
        opacity: 0,
        duration: 0.8,
        stagger: 0.15,
        delay: 0.4,
        ease: "power3.out",
      });
      gsap.from(".cta-btn", { y: 20, opacity: 0, duration: 0.6, delay: 0.9, ease: "power2.out" });
      gsap.from(".stat-card", {
        y: 20,
        opacity: 0,
        duration: 0.6,
        stagger: 0.1,
        delay: 1.1,
        ease: "power2.out",
      });
    });
    return () => ctx.revert();
  }, []);

  return (
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 w-full flex flex-col items-center">
      <section ref={heroRef} className="relative pt-12 pb-16 text-center w-full max-w-5xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-900/20 border border-red-500/30 text-red-400 text-xs font-mono mb-8">
          <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
          Modelo v7 · EfficientNetB0 · 92.3% Accuracy
        </div>

        <h1 className="text-4xl md:text-7xl font-black mb-6 leading-tight text-white drop-shadow-xl">
          CNN Geotermia
          <br />
          <span
            className="text-transparent bg-gradient-to-r from-red-500 via-orange-400 to-yellow-500"
            style={{ WebkitBackgroundClip: "text", backgroundClip: "text" }}
          >
            Colombia
          </span>
        </h1>
        <p className="text-xl md:text-2xl text-gray-300 font-light mb-16 max-w-3xl mx-auto">
          Identificación de zonas con potencial geotérmico mediante Deep Learning e imágenes satelitales ASTER de NASA
        </p>

        <div ref={cardsRef} className="grid md:grid-cols-3 gap-6 text-left mb-16">
          {[
            {
              Icon: Crosshair, color: "text-red-400", border: "hover:border-cyan-500/30",
              title: "Objetivo",
              desc: "Identificar zonas con alto potencial geotérmico en Colombia usando imágenes satelitales ASTER y redes neuronales convolucionales.",
            },
            {
              Icon: Layers, color: "text-orange-400", border: "hover:border-orange-500/30",
              title: "Datos",
              desc: "7 bandas ASTER (5 emisividad TIR + temperatura + NDVI), resolución 90 m, obtenidas vía Google Earth Engine para 407 zonas colombianas.",
            },
            {
              Icon: Cpu, color: "text-cyan-400", border: "hover:border-red-500/30",
              title: "Modelo",
              desc: "EfficientNetB0 con Channel Adapter (7→16→3), transfer learning en 2 fases, mixed precision float16 y 4.4M parámetros.",
            },
          ].map((card) => (
            <div key={card.title} className={`hero-card bg-[#12141c]/80 border border-white/5 p-8 rounded-3xl backdrop-blur-md ${card.border} transition-all duration-300 group`}>
              <div className={`mb-4 ${card.color} group-hover:scale-110 transition-transform`}>
                <card.Icon size={36} strokeWidth={1.5} />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">{card.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{card.desc}</p>
            </div>
          ))}
        </div>

        <Link
          to="/prediccion"
          className="cta-btn inline-flex items-center gap-3 bg-white text-black px-8 py-4 rounded-2xl font-bold text-lg transition-all shadow-[0_0_30px_rgba(255,255,255,0.1)] hover:shadow-[0_0_50px_rgba(255,255,255,0.25)] hover:scale-105 active:scale-95"
        >
          <Crosshair size={20} /> Probar el Escáner CNN
        </Link>
      </section>

      {/* Stats rápidas */}
      <section ref={statsRef} className="w-full max-w-5xl mx-auto py-16 border-t border-white/5">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          {[
            { label: "Accuracy", value: "92.3%", color: "text-green-400" },
            { label: "AUC-ROC", value: "97.4%", color: "text-red-400" },
            { label: "F1-Score", value: "92.2%", color: "text-purple-400" },
            { label: "Imágenes", value: "22,209", color: "text-cyan-400" },
          ].map((s) => (
            <div key={s.label} className="stat-card bg-white/5 rounded-2xl p-6 border border-white/5">
              <p className="text-[10px] text-gray-500 font-mono tracking-widest mb-2">{s.label.toUpperCase()}</p>
              <p className={`text-3xl font-black ${s.color}`}>{s.value}</p>
            </div>
          ))}
        </div>

        <div className="text-center mt-10">
          <Link
            to="/metricas"
            className="inline-flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors font-mono"
          >
            <BarChart3 size={16} /> Ver métricas detalladas →
          </Link>
        </div>
      </section>
    </div>
  );
}
