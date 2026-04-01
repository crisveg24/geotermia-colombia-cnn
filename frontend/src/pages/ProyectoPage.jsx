import React, { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { BookOpen, Crosshair } from "lucide-react";

gsap.registerPlugin(ScrollTrigger);

export default function ProyectoPage() {
  const aboutRef = useRef(null);
  const teamRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from(".about-item", {
        scrollTrigger: { trigger: aboutRef.current, start: "top 85%" },
        y: 30, opacity: 0, duration: 0.8, ease: "power2.out", stagger: 0.15,
      });

      // Animación profesional: fade-in escalonado con desplazamiento vertical suave
      gsap.from(".team-member", {
        scrollTrigger: { trigger: teamRef.current, start: "top 85%" },
        y: 40,
        opacity: 0,
        duration: 0.7,
        stagger: 0.12,
        ease: "power3.out",
      });

      gsap.from(".team-asesor", {
        scrollTrigger: { trigger: ".team-asesor", start: "top 90%" },
        y: 20,
        opacity: 0,
        duration: 0.8,
        delay: 0.4,
        ease: "power2.out",
      });
    });
    return () => ctx.revert();
  }, []);

  return (
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 w-full">
      {/* About */}
      <section ref={aboutRef} className="max-w-5xl mx-auto pt-12 mb-24">
        <div className="about-item text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-black text-white mb-4">Acerca del Proyecto</h2>
          <p className="text-gray-400">Proyecto de Grado · Universidad de San Buenaventura - Sede Bogotá</p>
          <p className="text-gray-500 text-sm font-mono mt-1">Programa: Ingeniería de Sistemas · 2025-2026</p>
        </div>

        <div className="about-item bg-[#12141c]/50 backdrop-blur-md rounded-3xl border border-white/5 p-8 mb-8">
          <h4 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
            <BookOpen size={20} className="text-cyan-400" /> Descripción
          </h4>
          <p className="text-gray-300 font-light leading-relaxed">
            Sistema de Deep Learning basado en CNN (EfficientNetB0) para la identificación automatizada de zonas con alto
            potencial geotérmico en Colombia, analizando imágenes satelitales térmicas del sensor NASA ASTER con 7 bandas
            espectrales (5 emisividad TIR + temperatura + NDVI). El modelo alcanza un 92.3% de accuracy y 97.4% AUC-ROC
            sobre un dataset de 22,209 imágenes de 407 zonas únicas.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <div className="about-item bg-[#12141c]/50 backdrop-blur-md rounded-3xl border border-white/5 p-8">
            <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Crosshair size={20} className="text-red-400" /> Objetivos
            </h4>
            <div className="mb-4">
              <strong className="text-gray-200 block mb-1">General:</strong>
              <p className="text-sm text-gray-400 font-light">
                Desarrollar un modelo predictivo de potencial geotérmico con visión por computador y deep learning.
              </p>
            </div>
            <div>
              <strong className="text-gray-200 block mb-2">Específicos:</strong>
              <ul className="text-sm text-gray-400 font-light list-disc list-inside space-y-2 ml-2">
                <li>Recopilar y procesar imágenes ASTER de zonas geotérmicas colombianas vía Google Earth Engine.</li>
                <li>Diseñar una arquitectura CNN basada en EfficientNetB0 con Channel Adapter para 7 bandas.</li>
                <li>Entrenar en 2 fases (frozen + fine-tuning) y evaluar con métricas estándar.</li>
                <li>Desarrollar interfaz web interactiva para visualización y predicción en tiempo real.</li>
              </ul>
            </div>
          </div>

          <div className="about-item bg-[#12141c]/50 backdrop-blur-md rounded-3xl border border-white/5 p-8 flex flex-col gap-6">
            <div>
              <h4 className="text-lg font-bold text-white mb-3">🛠️ Tecnologías</h4>
              <div className="space-y-4">
                <div>
                  <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">🧠 Deep Learning</span>
                  <p className="text-sm text-cyan-300 font-mono">TensorFlow 2.21 · Keras 3.12 · NumPy · scikit-learn</p>
                </div>
                <div>
                  <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">🌍 Geoespacial</span>
                  <p className="text-sm text-green-400 font-mono">Google Earth Engine · rasterio · geemap · GDAL</p>
                </div>
                <div>
                  <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">🖥️ Frontend</span>
                  <p className="text-sm text-orange-400 font-mono">React · Vite · Tailwind · Leaflet · Recharts · GSAP</p>
                </div>
                <div>
                  <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">⚙️ Backend</span>
                  <p className="text-sm text-purple-400 font-mono">Flask · Flask-CORS · Python 3.10</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="about-item bg-[#12141c]/30 backdrop-blur-md rounded-2xl border border-white/5 p-6 flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-gray-500">
          <div>
            <strong className="text-gray-300 block mb-1">Referencias Destacadas</strong>
            <ul className="list-disc list-inside text-xs leading-relaxed">
              <li>Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs.</li>
              <li>NASA ASTER Global Emissivity Dataset (AG100) V003.</li>
              <li>TensorFlow — tensorflow.org | Google Earth Engine — earthengine.google.com</li>
            </ul>
          </div>
          <div className="text-right">
            <span className="inline-block px-3 py-1 bg-white/5 border border-white/10 rounded-lg text-xs font-mono text-gray-400">
              Licencia MIT · Ver LICENSE
            </span>
          </div>
        </div>
      </section>

      {/* Team */}
      <section ref={teamRef} className="max-w-6xl mx-auto mb-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-5xl font-black text-white mb-4">Equipo de Investigación</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {[
            { id: "CV", name: "Cristian Camilo Vega S.", role: "Lead Developer", email: "ccvegas@academia.usbbog.edu.co" },
            { id: "DA", name: "Daniel Santiago Arévalo R.", role: "Co-autor", email: "dsarevalor@academia.usbbog.edu.co" },
            { id: "YE", name: "Yuliet Katerin Espitia A.", role: "Co-autora", email: "ykespitiaa@academia.usbbog.edu.co" },
            { id: "LR", name: "Laura Sophie Rivera M.", role: "Co-autora", email: "lsriveram@academia.usbbog.edu.co" },
          ].map((m) => (
            <div
              key={m.id}
              className="team-member bg-[#12141c]/50 backdrop-blur-md border border-white/5 p-6 rounded-3xl flex flex-col items-center text-center shadow-lg hover:bg-white/5 transition-colors"
            >
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-xl font-bold text-white mb-4 shadow-[0_0_15px_rgba(6,182,212,0.4)]">
                {m.id}
              </div>
              <h4 className="font-bold text-white mb-1">{m.name}</h4>
              <p className="text-cyan-400 text-xs uppercase tracking-wider mb-4 font-mono">{m.role}</p>
              <p className="text-gray-400 text-[11px] bg-black/30 px-3 py-1.5 rounded-full break-all w-full">{m.email}</p>
            </div>
          ))}
        </div>

        <div className="team-asesor bg-gradient-to-r from-transparent via-[#12141c] to-transparent border-y border-white/5 p-8 text-center mt-12">
          <p className="text-white text-lg font-medium mb-3">Asesor: Prof. Yeison Eduardo Conejo Sandoval</p>
          <p className="text-gray-500 font-mono text-xs tracking-widest uppercase">
            Universidad de San Buenaventura, Bogotá · Ingeniería de Sistemas · 2025-2026
          </p>
        </div>
      </section>
    </div>
  );
}
