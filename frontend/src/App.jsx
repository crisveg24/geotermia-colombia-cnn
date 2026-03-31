import React, { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Activity, Fingerprint, Database, Cpu, Radio, Flame, Crosshair, Map as MapIcon, MousePointerClick, Navigation, MapPin, Search, BarChart3, Layers, BookOpen } from "lucide-react";
import { MapContainer, TileLayer, Marker, CircleMarker, Popup, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Recharts
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, Legend } from 'recharts';

gsap.registerPlugin(ScrollTrigger);

// Leaflet icon fix
import icon from "leaflet/dist/images/marker-icon.png";
import iconShadow from "leaflet/dist/images/marker-shadow.png";

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

// Custom animated icon for the selection
const customIcon = new L.DivIcon({
  className: "custom-leaflet-icon",
  html: `<div class="w-6 h-6 bg-blue-500 rounded-full border-4 border-white shadow-[0_0_15px_rgba(59,130,246,1)] flex items-center justify-center">
            <div class="absolute w-12 h-12 bg-blue-500 rounded-full animate-ping opacity-50"></div>
         </div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

// Puntos Históricos / Zonas Geotérmicas evaluadas (De tu backend Python)
const zonasGeotermicas = [
  { nombre: "Nevado del Ruiz", lat: 4.8951, lon: -75.3222, potencial: "Alto" },
  { nombre: "Nevado del Tolima", lat: 4.6500, lon: -75.3667, potencial: "Alto" },
  { nombre: "Volcan Purace", lat: 2.3206, lon: -76.4036, potencial: "Alto" },
  { nombre: "Volcan Galeras", lat: 1.2208, lon: -77.3581, potencial: "Alto" },
  { nombre: "Volcan Cumbal", lat: 0.9539, lon: -77.8792, potencial: "Medio" },
  { nombre: "Volcan Sotara", lat: 2.1083, lon: -76.5917, potencial: "Medio" },
  { nombre: "Volcan Azufral", lat: 1.0833, lon: -77.7167, potencial: "Medio" },
  { nombre: "Paipa-Iza", lat: 5.7781, lon: -73.1124, potencial: "Alto" },
  { nombre: "Santa Rosa de Cabal", lat: 4.8694, lon: -75.6219, potencial: "Medio" },
  { nombre: "Manizales", lat: 5.0667, lon: -75.5167, potencial: "Medio" }
];

function MapClickEvents({ setCoords, mode }) {
  useMapEvents({
    click(e) {
      if (mode === "map") setCoords({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return null;
}

const barChartData = [
  { name: 'Accuracy', value: 68.4, color: '#f97316' },
  { name: 'Precision', value: 86.3, color: '#14b8a6' },
  { name: 'Recall', value: 48.1, color: '#3b82f6' },
  { name: 'F1-Score', value: 61.8, color: '#8b5cf6' },
];

const rocData = [
  { pfp: 0, ptp: 0, baseline: 0 },
  { pfp: 0.1, ptp: 0.35, baseline: 0.1 },
  { pfp: 0.2, ptp: 0.55, baseline: 0.2 },
  { pfp: 0.4, ptp: 0.72, baseline: 0.4 },
  { pfp: 0.6, ptp: 0.82, baseline: 0.6 },
  { pfp: 0.8, ptp: 0.93, baseline: 0.8 },
  { pfp: 1, ptp: 1, baseline: 1 },
];

const trainingHistoryData = [
  { epoch: 1, acc: 0.52, val_acc: 0.51, loss: 0.69, val_loss: 0.69 },
  { epoch: 15, acc: 0.68, val_acc: 0.64, loss: 0.55, val_loss: 0.58 },
  { epoch: 30, acc: 0.77, val_acc: 0.71, loss: 0.42, val_loss: 0.48 },
  { epoch: 45, acc: 0.85, val_acc: 0.76, loss: 0.31, val_loss: 0.41 },
  { epoch: 60, acc: 0.90, val_acc: 0.80, loss: 0.22, val_loss: 0.35 },
  { epoch: 80, acc: 0.94, val_acc: 0.81, loss: 0.15, val_loss: 0.33 },
  { epoch: 100, acc: 0.96, val_acc: 0.82, loss: 0.09, val_loss: 0.34 },
];

function App() {
  const titleRef = useRef(null);
  const introRef = useRef(null);
  const predictRef = useRef(null);
  const metricsRef = useRef(null);
  const chartsRef = useRef(null);
  const historyChartsRef = useRef(null);
  const archRef = useRef(null);
  const hyperRef = useRef(null);
  const aboutRef = useRef(null);
  const teamRef = useRef(null);

  const [inputMode, setInputMode] = useState("map");
  const [coords, setCoords] = useState({ lat: 4.5709, lng: -74.2973 });
  
  // States para la predicción real
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [zonaCercana, setZonaCercana] = useState(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Configurar que TODAS las animaciones de ScrollTrigger se reproduzcan al bajar y se reversen al subir
      ScrollTrigger.defaults({
        toggleActions: "play reverse play reverse"
      });

      gsap.from(introRef.current, { y: 60, opacity: 0, duration: 1.5, ease: "power4.out" });
      gsap.from(titleRef.current, { scrollTrigger: { trigger: titleRef.current, start: "top 80%" }, y: 80, opacity: 0, duration: 1.5, ease: "power4.out" });
      gsap.to(".parallax-satelite", { y: "250vh", rotation: 20, x: 50, ease: "none", scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: 1.5 }});
      gsap.fromTo(".parallax-magma", { y: "150vh" }, { y: "-50vh", x: -50, ease: "none", scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: 1.5 }});

      gsap.from(predictRef.current, { scrollTrigger: { trigger: predictRef.current, start: "top 85%" }, y: 100, opacity: 0, duration: 1.2, ease: "power3.out" });
      gsap.from(".card-stat", { scrollTrigger: { trigger: metricsRef.current, start: "top 80%" }, y: 60, opacity: 0, stagger: 0.15, duration: 1, ease: "back.out(1.5)" });
      gsap.from(".chart-box", { scrollTrigger: { trigger: chartsRef.current, start: "top 75%" }, scale: 0.9, opacity: 0, duration: 1, ease: "power2.out", stagger: 0.2 });
      
      gsap.from(".history-box", { scrollTrigger: { trigger: historyChartsRef.current, start: "top 75%" }, y: 50, opacity: 0, duration: 1, ease: "power2.out", stagger: 0.2 });

      gsap.from(archRef.current, { scrollTrigger: { trigger: archRef.current, start: "top 80%" }, y: 50, opacity: 0, duration: 1, ease: "power3.out" });
      
      gsap.from(".hyper-card", { scrollTrigger: { trigger: hyperRef.current, start: "top 80%" }, x: -50, opacity: 0, duration: 1, ease: "power3.out", stagger: 0.3 });
      
      gsap.from(".about-item", { scrollTrigger: { trigger: aboutRef.current, start: "top 85%" }, y: 30, opacity: 0, duration: 0.8, ease: "power2.out", stagger: 0.2 });

      // Animación de Equipo con Timeline MUY detallada
      var tl = gsap.timeline({
        scrollTrigger: {
          trigger: teamRef.current,
          start: "top 90%",
          end: "bottom 80%",
          scrub: 1.5 // El scrub mapea la animación exactamente a la posición de la barra de scroll
        }
      });
      
      // Sequenced one-after-the-other como solicitaste, con rotaciones, elastic y expo.
      const members = gsap.utils.toArray(".team-member");
      if (members.length >= 4) {
        tl.from(members[0], {rotation: -360, opacity: 0, scale: 0})
          .from(members[1], {x: -100, opacity: 0, ease: 'none'})
          .from(members[2], {rotation: 360, x: 100, opacity: 0, ease: 'none'})
          .from(members[3], {y: 100, opacity: 0, ease: 'none'})
          .from(".team-asesor", {scale: 0, rotationX: 180, opacity: 0, ease: 'none'});
      }

    });
    return () => ctx.revert();
  }, []);

  const handlePredict = async () => {
    setLoading(true); 
    setPrediction(null);
    setZonaCercana(null);
    
    try {
      // Usando el backend real de python
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat: coords.lat, lon: coords.lng })
      });
      
      const data = await response.json();
      
      const predValue = { val: 0 };
      gsap.to(predValue, {
        val: data.porcentaje, 
        duration: 1.5, 
        ease: "circ.out",
        onUpdate: () => setPrediction(predValue.val.toFixed(1)),
        onComplete: () => {
           setLoading(false);
           setZonaCercana(data.zona_cercana);
        }
      });
    } catch (error) {
      console.error("No se pudo conectar a la API", error);
      setLoading(false);
      setPrediction("ERROR");
    }
  };

  return (
    <div className="bg-[#0b0c10] text-gray-100 min-h-screen font-sans overflow-x-hidden relative selection:bg-red-500 selection:text-white">
      
      {/* Fondo Global */}
      <div className="fixed inset-0 pointer-events-none z-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/10 via-[#0b0c10] to-[#0b0c10]"></div>

      {/* Parallax Laterales */}
      <div className="parallax-satelite fixed top-10 left-4 md:left-12 z-0 opacity-10 pointer-events-none text-cyan-500 flex flex-col items-center">
         <Radio size={56} />
         <div className="h-64 w-[1px] bg-gradient-to-b from-cyan-500 to-transparent mt-4"></div>
      </div>
      <div className="parallax-magma fixed bottom-10 right-4 md:right-12 z-0 opacity-10 pointer-events-none text-orange-500 flex flex-col items-center">
         <div className="h-64 w-[1px] bg-gradient-to-t from-orange-500 to-transparent mb-4"></div>
         <Flame size={56} />
      </div>

      <nav className="relative w-full flex justify-center items-center py-8 z-50">
        <div className="flex items-center gap-3 font-bold text-2xl tracking-widest text-white uppercase backdrop-blur-md px-6 py-2 rounded-full bg-white/5 border border-white/10 shadow-[0_0_20px_rgba(0,0,0,0.5)]">
          <Activity className="text-red-500" /> Geotermia<span className="text-gray-500 font-light">CNN</span>
        </div>
      </nav>

      <div className="relative z-10 max-w-[1400px] mx-auto px-6 md:px-12 w-full flex flex-col items-center">
        
        {/* ================= HERO INTRO ================= */}
        <section ref={introRef} className="relative pt-8 pb-16 text-center w-full max-w-5xl mx-auto">
          <h1 className="text-4xl md:text-6xl font-black mb-6 leading-tight text-white drop-shadow-xl">
            CNN Geotermia Colombia
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 font-light mb-12">
            Identificación de zonas con potencial geotérmico mediante Deep Learning e imágenes satelitales ASTER
          </p>

          <div className="grid md:grid-cols-3 gap-6 text-left">
            <div className="bg-[#12141c]/80 border border-white/5 p-8 rounded-3xl backdrop-blur-md hover:border-cyan-500/30 transition-all">
              <div className="text-4xl mb-4">🎯</div>
              <h3 className="text-xl font-bold text-white mb-3">Objetivo</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Identificar zonas con alto potencial geotérmico en Colombia usando imágenes satelitales ASTER y redes neuronales convolucionales.
              </p>
            </div>

            <div className="bg-[#12141c]/80 border border-white/5 p-8 rounded-3xl backdrop-blur-md hover:border-orange-500/30 transition-all">
              <div className="text-4xl mb-4">🛰️</div>
              <h3 className="text-xl font-bold text-white mb-3">Datos</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Bandas térmicas NASA ASTER (10-14), resolución 100 m, obtenidas vía Google Earth Engine para zonas volcánicas colombianas.
              </p>
            </div>

            <div className="bg-[#12141c]/80 border border-white/5 p-8 rounded-3xl backdrop-blur-md hover:border-red-500/30 transition-all">
              <div className="text-4xl mb-4">🧠</div>
              <h3 className="text-xl font-bold text-white mb-3">Modelo</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                CNN ResNet-inspired con SpatialDropout2D, AdamW optimizer, Label Smoothing y ~5 millones de parámetros entrenables.
              </p>
            </div>
          </div>
        </section>

        {/* ================= TITULO ESCANER ================= */}
        <section className="relative pt-12 pb-16 text-center w-full border-t border-white/5 mt-8">
          <div ref={titleRef} className="max-w-4xl mx-auto flex flex-col items-center">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-900/30 border border-cyan-500/30 text-cyan-400 text-xs font-mono mb-6">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span> Backend de Python Conectado
            </div>
            <h1 className="text-5xl md:text-6xl font-black mb-6 leading-tight drop-shadow-2xl text-white">
              Escáner Térmico <br/> 
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-orange-400 to-yellow-500">
                Interactivo CNN
              </span>
            </h1>
            <p className="text-gray-400 text-lg font-light max-w-2xl">
               🗺️ Zonas Geotérmicas de Estudio. Selecciona libremente cualquier zona en la cordillera colombiana.
            </p>
          </div>
        </section>

        {/* ================= DASHBOARD PREDICCION ================= */}
        <section ref={predictRef} className="w-full mb-32 relative">
          
          <div className="flex flex-col xl:flex-row gap-8 bg-[#12141c]/80 backdrop-blur-xl border border-white/5 p-6 md:p-8 rounded-[2.5rem] shadow-[0_0_50px_rgba(0,0,0,0.8)] mx-auto w-full">
            
            {/* --- MAPA --- */}
            <div className="w-full xl:w-2/3 flex flex-col pb-4">
              <div className="flex flex-wrap gap-3 mb-6 justify-center md:justify-start">
                 <button onClick={() => setInputMode("map")} className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-all text-xs font-medium uppercase tracking-wider ${inputMode === 'map' ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400' : 'bg-transparent border-white/10 text-gray-500 hover:border-white/30'}`}>
                    <MousePointerClick size={14} /> Clic en Mapa
                 </button>
                 <button onClick={() => setInputMode("coords")} className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-all text-xs font-medium uppercase tracking-wider ${inputMode === 'coords' ? 'bg-cyan-500/20 border-cyan-500 text-cyan-400' : 'bg-transparent border-white/10 text-gray-500 hover:border-white/30'}`}>
                    <Navigation size={14} /> Coordenadas
                 </button>
                 <button onClick={() => setInputMode("zone")} className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-all text-xs font-medium uppercase tracking-wider ${inputMode === 'zone' ? 'bg-red-500/20 border-red-500 text-red-400' : 'bg-transparent border-white/10 text-gray-500 hover:border-white/30'}`}>
                    <MapPin size={14} /> Volcanes Activos
                 </button>
              </div>

              <div className="h-[550px] w-full relative rounded-3xl overflow-hidden border border-white/5 bg-white shadow-[inset_0_0_20px_rgba(0,0,0,0.2)]">
                <MapContainer center={[4.5709, -74.2973]} zoom={5.5} className="custom-map" style={{ height: '100%', width: '100%', cursor: inputMode === 'map' ? 'crosshair' : 'default' }}>
                    {/* CartoDB Positron BASE (Mismo fondo claro de tu imagen para que se vean los puntos igual) */}
                    <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" attribution="&copy; OpenStreetMap contributors &copy; CARTO" />
                    <MapClickEvents setCoords={setCoords} mode={inputMode} />
                    
                    {/* Zonas de Análisis Históricas */}
                    {zonasGeotermicas.map((zona, idx) => (
                      <CircleMarker 
                        key={idx}
                        center={[zona.lat, zona.lon]}
                        pathOptions={{ 
                          color: zona.potencial === 'Alto' ? '#ef4444' : '#f59e0b',
                          fillColor: zona.potencial === 'Alto' ? '#ef4444' : '#f59e0b',
                          fillOpacity: 0.6,
                          weight: 2
                        }}
                        radius={9}
                      >
                         <Popup className="font-mono text-xs">
                           <strong className="text-sm">{zona.nombre}</strong><br/>
                           Potencial: <span className={zona.potencial === 'Alto' ? 'text-red-600 font-bold' : 'text-orange-500 font-bold'}>{zona.potencial}</span>
                         </Popup>
                      </CircleMarker>
                    ))}

                    {/* Marcador de Selección */}
                    <Marker position={[coords.lat, coords.lng]} icon={customIcon} />
                </MapContainer>
                
                {/* LEYENDA DEL MAPA (Réplica a la foto del usuario) */}
                <div className="absolute bottom-6 left-6 z-[400] bg-white text-gray-800 p-4 rounded-xl shadow-[0_4px_15px_rgba(0,0,0,0.3)] border border-gray-200">
                    <h5 className="font-bold text-sm mb-3">Leyenda</h5>
                    <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 rounded-full bg-red-500 border border-red-700 opacity-80"></div><span className="text-xs">Potencial Alto</span></div>
                    <div className="flex items-center gap-2 mb-2"><div className="w-3 h-3 rounded-full bg-orange-400 border border-orange-600 opacity-80"></div><span className="text-xs">Potencial Medio</span></div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-blue-500 border border-blue-700 relative"><div className="absolute w-full h-full bg-blue-500 rounded-full animate-ping opacity-50"></div></div><span className="text-xs">Tu seleccion</span></div>
                </div>
              </div>
            </div>

            {/* --- PANEL DE ACCION --- */}
            <div className="w-full xl:w-1/3 flex flex-col gap-6">
              
              <div className="bg-[#1a1d27] rounded-[2rem] p-8 border border-white/5 relative overflow-hidden group">
                 <h3 className="text-sm font-mono tracking-widest text-gray-400 mb-6 border-b border-white/5 pb-4">PANEL DE INFERENCIA</h3>
                 
                 {inputMode === 'coords' || inputMode === 'map' ? (
                   <div className="grid grid-cols-2 gap-4 mb-8">
                     <div className="bg-[#0b0c10] rounded-xl p-4 border border-white/5 relative">
                         <label className="block text-[10px] tracking-widest font-mono text-gray-600 mb-1">LATITUD</label>
                         <input type="number" step="any" value={coords.lat} onChange={(e) => inputMode === 'coords' && setCoords({...coords, lat: e.target.value})} disabled={inputMode === 'map'} className="w-full bg-transparent text-xl text-white font-medium focus:outline-none focus:text-cyan-400 disabled:opacity-80" />
                     </div>
                     <div className="bg-[#0b0c10] rounded-xl p-4 border border-white/5 relative">
                         <label className="block text-[10px] tracking-widest font-mono text-gray-600 mb-1">LONGITUD</label>
                         <input type="number" step="any" value={coords.lng} onChange={(e) => inputMode === 'coords' && setCoords({...coords, lng: e.target.value})} disabled={inputMode === 'map'} className="w-full bg-transparent text-xl text-white font-medium focus:outline-none focus:text-cyan-400 disabled:opacity-80" />
                     </div>
                   </div>
                 ) : (
                   <div className="mb-8 p-4 bg-[#0b0c10] rounded-xl border border-white/5">
                     <label className="block text-[10px] tracking-widest font-mono text-gray-600 mb-2">PUNTO DE INTERÉS</label>
                      <select onChange={(e) => {
                        const [lat, lng] = e.target.value.split(',');
                        setCoords({lat: parseFloat(lat), lng: parseFloat(lng)});
                      }} className="w-full bg-transparent text-white font-medium focus:outline-none text-base p-2">
                         <option className="bg-[#1a1d27]" value="4.8951,-75.3222">Nevado del Ruiz</option>
                         <option className="bg-[#1a1d27]" value="2.3206,-76.4036">Volcán Puracé</option>
                         <option className="bg-[#1a1d27]" value="5.7781,-73.1124">Sistema Paipa</option>
                      </select>
                   </div>
                 )}

                 <button onClick={handlePredict} disabled={loading} className="w-full bg-white text-black py-4 rounded-xl font-bold transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.3)] flex justify-center items-center gap-2 active:scale-[0.98]">
                   {loading ? <span className="animate-spin border-2 border-black/30 border-t-black w-5 h-5 rounded-full"></span> : <><Database size={18}/> COMPUTAR MODELO</>}
                 </button>
              </div>

              <div className="flex-grow bg-[#1a1d27] rounded-[2rem] p-6 border border-white/5 flex justify-center items-center text-center relative overflow-hidden">
                 <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-red-900/10 via-transparent to-transparent opacity-50 mix-blend-screen pointer-events-none"></div>
                 {prediction !== null && !loading && prediction !== "ERROR" ? (
                     <div className="relative z-10 animate-fade-in w-full">
                       <p className="text-[10px] font-mono text-gray-500 tracking-[0.3em] uppercase mb-4">Potencial Calculado</p>
                       <div className={`text-6xl md:text-7xl font-black text-transparent bg-clip-text drop-shadow-[0_0_15px_rgba(255,255,255,0.1)] mb-6 ${parseFloat(prediction) > 70 ? 'bg-gradient-to-br from-white via-orange-200 to-red-500' : parseFloat(prediction) > 30 ? 'bg-gradient-to-br from-white via-yellow-200 to-orange-500' : 'bg-gradient-to-br from-white via-gray-200 to-gray-500'}`}>
                         {prediction}%
                       </div>
                       
                       <div className="flex flex-col items-center gap-2">
                         {parseFloat(prediction) >= 70 ? (
                           <div className="inline-flex items-center gap-2 bg-red-500/10 border border-red-500/30 px-4 py-2 rounded-full"><span className="w-2 h-2 rounded-full bg-red-500 animate-ping"></span><span className="text-xs font-bold text-red-500">POTENCIAL ALTO</span></div>
                         ) : parseFloat(prediction) >= 30 ? (
                           <div className="inline-flex items-center gap-2 bg-orange-500/10 border border-orange-500/30 px-4 py-2 rounded-full"><span className="w-2 h-2 rounded-full bg-orange-500"></span><span className="text-xs font-bold text-orange-500">POTENCIAL MEDIO</span></div>
                         ) : (
                           <div className="inline-flex items-center gap-2 bg-gray-500/10 border border-gray-500/30 px-4 py-2 rounded-full"><span className="w-2 h-2 rounded-full bg-gray-400"></span><span className="text-xs font-bold text-gray-400">POTENCIAL BAJO</span></div>
                         )}

                         {zonaCercana && (
                           <p className="text-xs text-gray-400 mt-2 font-light">Calculado desde ref: <span className="text-white font-medium">{zonaCercana}</span></p>
                         )}
                       </div>
                     </div>
                 ) : prediction === "ERROR" ? (
                      <div className="text-red-500"><Activity size={40} className="mx-auto mb-4"/> <span className="text-xs font-mono">Error de red API Python</span></div>
                 ) : loading ? (
                     <div className="text-cyan-500"><Cpu size={40} className="mx-auto mb-4 animate-bounce"/> <span className="text-[10px] font-mono tracking-widest animate-pulse">EVALUANDO TENSOR_</span></div>
                 ) : (
                     <div className="opacity-20"><MapIcon size={40} strokeWidth={1} className="mx-auto mb-4"/> <span className="text-[10px] uppercase font-mono tracking-widest">Esperando Ubicación</span></div>
                 )}
              </div>

            </div>
          </div>
        </section>

        {/* ================= METRICAS SECCIÓN (Scroll Animated) ================= */}
        <section className="w-full max-w-6xl mx-auto mb-32">
          
          <div className="text-center mb-16">
             <h2 className="text-4xl md:text-5xl font-black text-white mb-4">Evaluación del Modelo</h2>
             <p className="text-gray-500 font-light text-lg">Métricas finales reportadas durante la de fase testing.</p>
          </div>

          <div ref={metricsRef} className="grid grid-cols-2 lg:grid-cols-5 gap-4 lg:gap-6 mb-16">
             <div className="card-stat bg-white/5 backdrop-blur rounded-[1.5rem] p-6 border border-white/10 flex flex-col items-center">
                 <span className="text-[10px] text-gray-500 font-mono mb-2 tracking-widest">ACCURACY</span>
                 <span className="text-4xl font-black text-[#f97316]">68.4%</span>
             </div>
             <div className="card-stat bg-white/5 backdrop-blur rounded-[1.5rem] p-6 border border-white/10 flex flex-col items-center">
                 <span className="text-[10px] text-gray-500 font-mono mb-2 tracking-widest">PRECISION</span>
                 <span className="text-4xl font-black text-[#14b8a6]">86.3%</span>
             </div>
             <div className="card-stat bg-white/5 backdrop-blur rounded-[1.5rem] p-6 border border-white/10 flex flex-col items-center">
                 <span className="text-[10px] text-gray-500 font-mono mb-2 tracking-widest">RECALL</span>
                 <span className="text-4xl font-black text-[#3b82f6]">48.1%</span>
             </div>
             <div className="card-stat bg-white/5 backdrop-blur rounded-[1.5rem] p-6 border border-white/10 flex flex-col items-center">
                 <span className="text-[10px] text-gray-500 font-mono mb-2 tracking-widest">F1-SCORE</span>
                 <span className="text-4xl font-black text-[#8b5cf6]">61.8%</span>
             </div>
             <div className="card-stat bg-white/5 backdrop-blur rounded-[1.5rem] p-6 border border-white/10 flex flex-col items-center col-span-2 lg:col-span-1 shadow-[0_0_30px_rgba(239,68,68,0.15)] border-red-500/30">
                 <span className="text-[10px] text-red-500/80 font-mono mb-2 tracking-widest">AUC-ROC</span>
                 <span className="text-4xl font-black text-[#ef4444]">82.0%</span>
             </div>
          </div>

          <div ref={chartsRef} className="grid md:grid-cols-2 gap-8 w-full">
              {/* Box 1 */}
              <div className="chart-box bg-white rounded-[2rem] p-8 shadow-2xl relative overflow-visible w-full min-h-[400px]">
                 <h4 className="text-xl font-bold text-gray-800 mb-8 border-b border-gray-100 pb-4">Comparativa de Métricas</h4>
                 <div className="h-[280px] w-full text-black">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={barChartData} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#6b7280', fontSize: 12}} dy={10} />
                        <YAxis tickFormatter={(val) => `${val}%`} axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
                        <Tooltip cursor={{fill: '#f3f4f6'}} contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}} />
                        <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={60}>
                          {barChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                 </div>
              </div>

              {/* Box 2 */}
              <div className="chart-box bg-white rounded-[2rem] p-8 shadow-2xl relative overflow-visible w-full min-h-[400px]">
                 <h4 className="text-xl font-bold text-gray-800 mb-8 border-b border-gray-100 pb-4">Curva ROC</h4>
                 <div className="h-[280px] w-full text-black">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={rocData} margin={{ top: 10, right: 30, left: -10, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="pfp" type="number" domain={[0, 1]} tickCount={6} axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} dy={10} />
                        <YAxis domain={[0, 1]} tickCount={6} axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}/>
                        <Legend verticalAlign="bottom" height={36} iconType="plainline" />
                        <Line type="monotone" dataKey="ptp" name="ROC (AUC = 0.820)" stroke="#ef4444" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                        <Line type="linear" dataKey="baseline" name="Aleatorio" stroke="#d1d5db" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                 </div>
              </div>
          </div>

          <div ref={historyChartsRef} className="grid md:grid-cols-2 gap-8 w-full mt-8">
              {/* Box Historial Accuracy */}
              <div className="history-box bg-white rounded-[2rem] p-8 shadow-2xl relative overflow-visible w-full min-h-[400px]">
                 <h4 className="text-xl font-bold text-gray-800 mb-8 border-b border-gray-100 pb-4">Historial de Entrenamiento (Accuracy)</h4>
                 <div className="h-[280px] w-full text-black">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingHistoryData} margin={{ top: 10, right: 30, left: -10, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="epoch" type="number" domain={[1, 100]} tickCount={6} axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} dy={10} name="Época" />
                        <YAxis domain={[0, 1]} tickCount={6} axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}/>
                        <Legend verticalAlign="bottom" height={36} iconType="plainline" />
                        <Line type="monotone" dataKey="acc" name="Train Acc" stroke="#f97316" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                        <Line type="monotone" dataKey="val_acc" name="Val Acc" stroke="#8b5cf6" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                      </LineChart>
                    </ResponsiveContainer>
                 </div>
              </div>

              {/* Box Historial Loss */}
              <div className="history-box bg-white rounded-[2rem] p-8 shadow-2xl relative overflow-visible w-full min-h-[400px]">
                 <h4 className="text-xl font-bold text-gray-800 mb-8 border-b border-gray-100 pb-4">Historial de Entrenamiento (Loss)</h4>
                 <div className="h-[280px] w-full text-black">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingHistoryData} margin={{ top: 10, right: 30, left: -10, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="epoch" type="number" domain={[1, 100]} tickCount={6} axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} dy={10} name="Época" />
                        <YAxis domain={[0, 'auto']} axisLine={false} tickLine={false} tick={{fill: '#9ca3af', fontSize: 12}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}/>
                        <Legend verticalAlign="bottom" height={36} iconType="plainline" />
                        <Line type="monotone" dataKey="loss" name="Train Loss" stroke="#ef4444" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                        <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#3b82f6" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                      </LineChart>
                    </ResponsiveContainer>
                 </div>
              </div>
          </div>
        </section>

        {/* ================= ARQUITECTURA SECCIÓN ================= */}
        <section ref={archRef} className="w-full max-w-5xl mx-auto mb-32">
          
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-5xl font-black text-white mb-4">Arquitectura del Modelo CNN</h2>
          </div>

          <div className="bg-[#12141c]/50 backdrop-blur-md border border-white/5 border-l-4 border-l-red-500 rounded-2xl p-6 md:p-8 mb-12 shadow-2xl">
             <p className="font-light text-gray-300 leading-relaxed text-lg text-center">
                Arquitectura <strong className="text-white">ResNet-inspired</strong> con bloques residuales, SpatialDropout2D para regularización espacial y Global Average Pooling. Diseñada para clasificación binaria de imágenes térmicas ASTER de 5 bandas.
             </p>
          </div>

          <div className="bg-[#12141c] backdrop-blur-md rounded-[2rem] border border-white/5 shadow-2xl p-2">
             <table className="w-full text-left border-collapse rounded-[1.8rem] overflow-hidden">
                <thead>
                   <tr className="bg-[#1a1d27] uppercase text-[10px] tracking-widest text-gray-500">
                      <th className="p-6 font-medium">Capa / Componente</th>
                      <th className="p-6 font-medium">Filtros</th>
                      <th className="p-6 font-medium text-right">Salida Tensor</th>
                   </tr>
                </thead>
                <tbody className="text-sm font-mono text-gray-300">
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 text-cyan-400 font-bold">Input Image Layer</td>
                      <td className="p-5 opacity-30">—</td>
                      <td className="p-5 pr-6 text-right">224x224x5</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 flex flex-col"><span className="text-white">Conv2D</span> <span className="text-[10px] text-gray-500 tracking-wider">BN + ReLU</span></td>
                      <td className="p-5 font-bold text-gray-400">32 <span className="text-xs font-normal opacity-50">(7x7)</span></td>
                      <td className="p-5 pr-6 text-right">224x224x32</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 text-purple-400">SpatialDropout2D</td>
                      <td className="p-5 opacity-30">—</td>
                      <td className="p-5 pr-6 text-right">224x224x32</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 text-orange-400">MaxPooling2D</td>
                      <td className="p-5 opacity-30">—</td>
                      <td className="p-5 pr-6 text-right font-bold text-white">112x112x32</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/10 transition-colors bg-green-500/5">
                      <td className="p-5 pl-6 text-green-400 font-bold border-l-2 border-green-500">Residual Block 1</td>
                      <td className="p-5 text-gray-200">64</td>
                      <td className="p-5 pr-6 text-right">112x112x64</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 flex flex-col"><span className="text-gray-300">SpatialDropout2D</span> <span className="text-[10px] text-gray-500 tracking-wider">MaxPool</span></td>
                      <td className="p-5 opacity-30">—</td>
                      <td className="p-5 pr-6 text-right font-bold text-white">56x56x64</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/10 transition-colors bg-green-500/5">
                      <td className="p-5 pl-6 text-green-400 font-bold border-l-2 border-green-500">Residual Block 2</td>
                      <td className="p-5 text-gray-200">128</td>
                      <td className="p-5 pr-6 text-right">56x56x128</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/10 transition-colors bg-green-500/5">
                      <td className="p-5 pl-6 text-green-400 font-bold border-l-2 border-green-500">Residual Block 3</td>
                      <td className="p-5 text-white bg-green-500/20 px-2 py-1 rounded inline-block">256</td>
                      <td className="p-5 pr-6 text-right">28x28x256</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 flex flex-col"><span className="text-gray-300">SpatialDropout2D</span> <span className="text-[10px] text-gray-500 tracking-wider">MaxPool</span></td>
                      <td className="p-5 opacity-30">—</td>
                      <td className="p-5 pr-6 text-right font-bold text-white">14x14x256</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/10 transition-colors bg-green-500/5">
                      <td className="p-5 pl-6 text-green-400 font-bold border-l-2 border-green-500">Residual Block 4</td>
                      <td className="p-5 text-white bg-green-500/20 px-2 py-1 rounded inline-block">512</td>
                      <td className="p-5 pr-6 text-right">14x14x512</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 text-purple-400">SpatialDropout2D</td>
                      <td className="p-5 opacity-30">—</td>
                      <td className="p-5 pr-6 text-right">14x14x512</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 text-yellow-500 font-bold">Global Average Pooling</td>
                      <td className="p-5 opacity-30">—</td>
                      <td className="p-5 pr-6 text-right text-white">512</td>
                   </tr>
                   <tr className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="p-5 pl-6 flex flex-col"><span className="text-teal-400">Dense</span> <span className="text-[10px] text-gray-500 tracking-wider">BN + Dropout</span></td>
                      <td className="p-5 opacity-30">—</td>
                      <td className="p-5 pr-6 text-right font-bold text-white">256</td>
                   </tr>
                   <tr className="border-t-2 border-red-500/50 hover:bg-red-500/10 transition-colors bg-red-500/5">
                      <td className="p-5 pl-6 text-red-500 font-black">Output</td>
                      <td className="p-5 text-red-400 text-xs font-mono">Sigmoid</td>
                      <td className="p-5 pr-6 text-right font-bold text-red-500 text-lg">1</td>
                   </tr>
                </tbody>
             </table>
          </div>
        </section>

        {/* ================= HIPERPARAMETROS Y DATOS DE ENTRADA ================= */}
        <section ref={hyperRef} className="w-full max-w-6xl mx-auto mb-32 grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="hyper-card bg-[#12141c]/80 backdrop-blur-md rounded-3xl border border-white/5 p-8 shadow-2xl relative overflow-hidden">
                <div className="absolute -right-10 -top-10 text-cyan-500 opacity-5">
                    <Cpu size={150} strokeWidth={1} />
                </div>
                <h3 className="text-2xl font-bold text-white mb-6 border-b border-white/10 pb-4 flex items-center gap-3">
                    <Database className="text-cyan-400" /> Datos de Entrada
                </h3>
                <ul className="flex flex-col gap-4 text-sm font-light text-gray-300">
                    <li className="flex justify-between items-center"><strong className="text-white">Fuente</strong> <span className="text-right max-w-[200px] font-mono text-xs">NASA ASTER (AG100) V003</span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Resolución</strong> <span className="font-mono text-xs">100 metros</span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Bandas</strong> <span className="font-mono text-xs text-cyan-400">10, 11, 12, 13, 14 (TIR)</span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Entrada Tensor</strong> <span className="font-mono text-xs font-bold text-white">224 x 224 x 5</span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Normalización</strong> <span className="font-mono text-xs">0-1 (Rescaling)</span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Dataset Total</strong> <span className="font-mono text-xs text-green-400">5,518 imágenes aug.</span></li>
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
                    <li className="flex justify-between items-center"><strong className="text-white">Optimizador</strong> <span className="text-right font-mono text-xs">AdamW <span className="opacity-50">(wd=1e-4)</span></span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Learning Rate</strong> <span className="font-mono text-xs text-orange-400 font-bold">0.001</span></li>
                    <li className="flex justify-between text-right gap-4"><strong className="text-white whitespace-nowrap">Función de Loss</strong> <span className="font-mono text-[11px]">BinaryCrossentropy <span className="opacity-50">(ls=0.1)</span></span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Dropout</strong> <span className="font-mono text-xs">0.5 (D) · 0.1-0.3 (S)</span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Batch Size</strong> <span className="font-mono text-xs text-white">32</span></li>
                    <li className="flex justify-between items-center"><strong className="text-white">Épocas</strong> <span className="font-mono text-xs">100 <span className="text-red-400">(EarlyStop p=15)</span></span></li>
                    <li className="flex justify-between items-center pt-2 border-t border-white/5"><strong className="text-white uppercase tracking-wider text-xs">Parámetros Entrenables</strong> <span className="font-mono text-sm text-yellow-400 font-bold">5,025,409</span></li>
                </ul>
            </div>
        </section>

        {/* ================= ACERCA DEL PROYECTO ================= */}
        <section ref={aboutRef} className="w-full max-w-5xl mx-auto mb-32">
            <div className="about-item text-center mb-12">
                <h2 className="text-3xl md:text-5xl font-black text-white mb-4">📋 Acerca del Proyecto</h2>
                <p className="text-gray-400">Proyecto de Grado · Universidad de San Buenaventura - Sede Bogotá</p>
                <p className="text-gray-500 text-sm font-mono mt-1">Programa: Ingeniería de Sistemas · 2025-2026</p>
            </div>

            <div className="about-item bg-[#12141c]/50 backdrop-blur-md rounded-3xl border border-white/5 p-8 mb-8">
                <h4 className="text-lg font-bold text-white mb-3 flex items-center gap-2"><BookOpen size={20} className="text-cyan-400"/> Descripción</h4>
                <p className="text-gray-300 font-light leading-relaxed">
                    Sistema de Deep Learning basado en CNN para la identificación automatizada de zonas con alto potencial geotérmico en Colombia, analizando imágenes satelitales térmicas del sensor NASA ASTER.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-8">
                <div className="about-item bg-[#12141c]/50 backdrop-blur-md rounded-3xl border border-white/5 p-8">
                    <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2"><Crosshair size={20} className="text-red-400"/> Objetivos</h4>
                    <div className="mb-4">
                        <strong className="text-gray-200 block mb-1">General:</strong>
                        <p className="text-sm text-gray-400 font-light">Desarrollar un modelo predictivo de potencial geotérmico con visión por computador y deep learning.</p>
                    </div>
                    <div>
                        <strong className="text-gray-200 block mb-2">Específicos:</strong>
                        <ul className="text-sm text-gray-400 font-light list-disc list-inside space-y-2 ml-2">
                            <li>Recopilar y procesar imágenes ASTER de zonas geotérmicas colombianas.</li>
                            <li>Diseñar una arquitectura CNN optimizada con bloques residuales.</li>
                            <li>Entrenar y evaluar con métricas estándar de clasificación.</li>
                            <li>Desarrollar interfaz web interactiva para visualización y predicción.</li>
                        </ul>
                    </div>
                </div>

                <div className="about-item bg-[#12141c]/50 backdrop-blur-md rounded-3xl border border-white/5 p-8 flex flex-col gap-6">
                    <div>
                        <h4 className="text-lg font-bold text-white mb-3">🛠️ Tecnologías</h4>
                        <div className="space-y-4">
                            <div>
                                <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">🧠 Deep Learning</span>
                                <p className="text-sm text-cyan-300 font-mono">TensorFlow 2.20 · Keras · NumPy · scikit-learn</p>
                            </div>
                            <div>
                                <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">🌍 Datos & Geo Espacial</span>
                                <p className="text-sm text-green-400 font-mono">Google Earth Engine · rasterio · pandas · GDAL</p>
                            </div>
                            <div>
                                <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">📊 Visualización</span>
                                <p className="text-sm text-orange-400 font-mono">Plotly · Folium · Matplotlib · Seaborn · Streamlit</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="about-item bg-[#12141c]/30 backdrop-blur-md rounded-2xl border border-white/5 p-6 flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-gray-500">
                <div>
                   <strong className="text-gray-300 block mb-1">Referencias Destacadas</strong>
                   <ul className="list-disc list-inside text-xs leading-relaxed">
                      <li>He, K. et al. (2016). Deep Residual Learning for Image Recognition.</li>
                      <li>NASA ASTER Global Emissivity Dataset (AG100) V003.</li>
                      <li>TensorFlow — tensorflow.org | Google Earth Engine — earthengine.google.com</li>
                   </ul>
                </div>
                <div className="text-right">
                    <span className="inline-block px-3 py-1 bg-white/5 border border-white/10 rounded-lg text-xs font-mono text-gray-400">Licencia MIT · Ver LICENSE</span>
                </div>
            </div>
        </section>

        {/* ================= EQUIPO DE INVESTIGACION ================= */}
        <section ref={teamRef} className="w-full max-w-6xl mx-auto mb-32">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-5xl font-black text-white mb-4">👥 Equipo de Investigación</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {[
              { id: "CV", name: "Cristian Camilo Vega S.", role: "Lead Developer", email: "ccvegas@academia.usbbog.edu.co" },
              { id: "DA", name: "Daniel Santiago Arevalo R.", role: "Co-autor", email: "dsarevalor@academia.usbbog.edu.co" },
              { id: "YE", name: "Yuliet Katerin Espitia A.", role: "Co-autora", email: "ykespitiaa@academia.usbbog.edu.co" },
              { id: "LR", name: "Laura Sophie Rivera M.", role: "Co-autora", email: "lsriveram@academia.usbbog.edu.co" }
            ].map(m => (
              <div key={m.id} className="team-member bg-[#12141c]/50 backdrop-blur-md border border-white/5 p-6 rounded-3xl flex flex-col items-center text-center shadow-lg hover:bg-white/5 transition-colors">
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
             <p className="text-gray-500 font-mono text-xs tracking-widest uppercase">Universidad de San Buenaventura, Bogotá · Ingeniería de Sistemas · 2025-2026</p>
          </div>
        </section>

      </div>
      
      <footer className="w-full text-center py-8 border-t border-white/5 bg-[#0b0c10] relative z-20">
         <p className="text-[10px] font-mono text-gray-600 tracking-[0.2em] mb-2">UNIVERSIDAD DE SAN BUENAVENTURA</p>
         <p className="text-[10px] font-mono text-gray-700 tracking-widest">INGENIERÍA DE SISTEMAS · IA APLICADA</p>
      </footer>

    </div>
  );
}

export default App;