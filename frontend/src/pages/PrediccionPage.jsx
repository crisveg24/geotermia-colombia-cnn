import React, { useEffect, useRef, useState, useCallback } from "react";
import { gsap } from "gsap";
import { MapContainer, TileLayer, Marker, CircleMarker, Popup, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Database, MousePointerClick, Navigation, MapPin, Cpu, Activity,
  Satellite, Layers, CheckCircle2, XCircle, Trash2, ThermometerSun,
} from "lucide-react";

// Leaflet icon fix
import icon from "leaflet/dist/images/marker-icon.png";
import iconShadow from "leaflet/dist/images/marker-shadow.png";
let DefaultIcon = L.icon({ iconUrl: icon, shadowUrl: iconShadow, iconSize: [25, 41], iconAnchor: [12, 41] });
L.Marker.prototype.options.icon = DefaultIcon;

const customIcon = new L.DivIcon({
  className: "custom-leaflet-icon",
  html: `<div class="w-6 h-6 bg-blue-500 rounded-full border-4 border-white shadow-[0_0_15px_rgba(59,130,246,1)] flex items-center justify-center">
            <div class="absolute w-12 h-12 bg-blue-500 rounded-full animate-ping opacity-50"></div>
         </div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

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
  { nombre: "Manizales", lat: 5.0667, lon: -75.5167, potencial: "Medio" },
];

const HISTORY_KEY = "geotermia_prediction_history";

function getHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, 10)));
}

function heatColor(pct) {
  if (pct >= 70) return "#ef4444";
  if (pct >= 50) return "#f97316";
  if (pct >= 30) return "#eab308";
  return "#6b7280";
}

function MapClickEvents({ setCoords, mode }) {
  useMapEvents({
    click(e) {
      if (mode === "map") setCoords({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return null;
}

/* ──────── Componente principal ──────── */
export default function PrediccionPage() {
  const sectionRef = useRef(null);
  const [inputMode, setInputMode] = useState("map");
  const [coords, setCoords] = useState({ lat: 4.5709, lng: -74.2973 });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [healthInfo, setHealthInfo] = useState(null);
  const [history, setHistory] = useState(getHistory);

  // Fetch model health on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then(setHealthInfo)
      .catch(() => setHealthInfo({ status: "error", modelo_cargado: "no disponible", modelo_disponible: false }));
  }, []);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from(sectionRef.current, { y: 40, opacity: 0, duration: 1, ease: "power3.out" });
    });
    return () => ctx.revert();
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    setPrediction(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat: coords.lat, lon: coords.lng }),
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${response.status}`);
      }
      const data = await response.json();
      setResult(data);

      const predValue = { val: 0 };
      gsap.to(predValue, {
        val: data.porcentaje,
        duration: 1.5,
        ease: "circ.out",
        onUpdate: () => setPrediction(predValue.val.toFixed(1)),
        onComplete: () => setLoading(false),
      });

      // Save to history (localStorage)
      const entry = {
        lat: coords.lat,
        lon: coords.lng,
        porcentaje: data.porcentaje,
        zona: data.zona_cercana,
        metodo: data.metodo,
        timestamp: new Date().toISOString(),
      };
      const updated = [entry, ...getHistory()].slice(0, 10);
      saveHistory(updated);
      setHistory(updated);
    } catch (error) {
      console.error("Error API:", error);
      setLoading(false);
      setPrediction("ERROR");
      setResult({ error: error.message });
    }
  };

  const clearHistory = useCallback(() => {
    localStorage.removeItem(HISTORY_KEY);
    setHistory([]);
  }, []);

  return (
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 w-full flex flex-col items-center">
      {/* Titulo */}
      <section className="relative pt-8 pb-8 text-center w-full max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-900/30 border border-cyan-500/30 text-cyan-400 text-xs font-mono mb-6">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          Backend de Python Conectado
        </div>
        <h1 className="text-4xl md:text-5xl font-black mb-4 leading-tight text-white">
          Escáner Térmico{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-orange-400 to-yellow-500">
            Interactivo CNN
          </span>
        </h1>
        <p className="text-gray-400 text-lg font-light">
          Selecciona una ubicación en Colombia para analizar su potencial geotérmico con el modelo EfficientNetB0.
        </p>
      </section>

      {/* ── Estado del Modelo ── */}
      <section className="w-full max-w-6xl mb-6">
        <div className="bg-[#12141c]/60 backdrop-blur-md rounded-2xl border border-white/5 px-6 py-4 flex flex-wrap gap-x-8 gap-y-3 items-center text-sm">
          <div className="flex items-center gap-2">
            {healthInfo?.modelo_disponible ? (
              <CheckCircle2 size={16} className="text-green-400" />
            ) : (
              <XCircle size={16} className="text-red-400" />
            )}
            <span className="text-gray-400">Modelo:</span>
            <span className="text-white font-mono text-xs">
              {healthInfo?.modelo_cargado || "cargando…"}
            </span>
          </div>
          {healthInfo?.modelo_detalle && (
            <>
              <div className="text-gray-500 text-xs font-mono">
                Input: <span className="text-cyan-400">{healthInfo.modelo_detalle.input_shape?.join(" × ") || "—"}</span>
              </div>
              <div className="text-gray-500 text-xs font-mono">
                Params: <span className="text-cyan-400">{healthInfo.modelo_detalle.total_params?.toLocaleString() || "—"}</span>
              </div>
            </>
          )}
          {healthInfo && !healthInfo.modelo_disponible && (
            <span className="text-orange-400 text-xs">Se usará fallback por proximidad</span>
          )}
        </div>
      </section>

      {/* ── Dashboard principal ── */}
      <section ref={sectionRef} className="w-full mb-16 relative">
        <div className="flex flex-col xl:flex-row gap-8 bg-[#12141c]/80 backdrop-blur-xl border border-white/5 p-6 md:p-8 rounded-[2.5rem] shadow-[0_0_50px_rgba(0,0,0,0.8)] mx-auto w-full">
          {/* MAPA */}
          <div className="w-full xl:w-2/3 flex flex-col pb-4">
            <div className="flex flex-wrap gap-3 mb-6 justify-center md:justify-start">
              {[
                { mode: "map", label: "Clic en Mapa", Icon: MousePointerClick, color: "cyan" },
                { mode: "coords", label: "Coordenadas", Icon: Navigation, color: "cyan" },
                { mode: "zone", label: "Volcanes Activos", Icon: MapPin, color: "red" },
              ].map(({ mode: m, label, Icon, color }) => (
                <button
                  key={m}
                  onClick={() => setInputMode(m)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-all text-xs font-medium uppercase tracking-wider ${
                    inputMode === m
                      ? color === "red"
                        ? "bg-red-500/20 border-red-500 text-red-400"
                        : "bg-cyan-500/20 border-cyan-500 text-cyan-400"
                      : "bg-transparent border-white/10 text-gray-500 hover:border-white/30"
                  }`}
                >
                  <Icon size={14} /> {label}
                </button>
              ))}
            </div>

            <div className="h-[550px] w-full relative rounded-3xl overflow-hidden border border-white/5 bg-white shadow-[inset_0_0_20px_rgba(0,0,0,0.2)]">
              <MapContainer
                center={[4.5709, -74.2973]}
                zoom={5.5}
                style={{ height: "100%", width: "100%", cursor: inputMode === "map" ? "crosshair" : "default" }}
              >
                <TileLayer
                  url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                  attribution="&copy; OpenStreetMap contributors &copy; CARTO"
                />
                <MapClickEvents setCoords={setCoords} mode={inputMode} />

                {zonasGeotermicas.map((zona, idx) => (
                  <CircleMarker
                    key={idx}
                    center={[zona.lat, zona.lon]}
                    pathOptions={{
                      color: zona.potencial === "Alto" ? "#ef4444" : "#f59e0b",
                      fillColor: zona.potencial === "Alto" ? "#ef4444" : "#f59e0b",
                      fillOpacity: 0.6,
                      weight: 2,
                    }}
                    radius={9}
                  >
                    <Popup className="font-mono text-xs">
                      <strong className="text-sm">{zona.nombre}</strong>
                      <br />
                      Potencial:{" "}
                      <span className={zona.potencial === "Alto" ? "text-red-600 font-bold" : "text-orange-500 font-bold"}>
                        {zona.potencial}
                      </span>
                    </Popup>
                  </CircleMarker>
                ))}

                <Marker position={[coords.lat, coords.lng]} icon={customIcon} />
              </MapContainer>

              {/* Leyenda */}
              <div className="absolute bottom-6 left-6 z-[400] bg-white text-gray-800 p-4 rounded-xl shadow-[0_4px_15px_rgba(0,0,0,0.3)] border border-gray-200">
                <h5 className="font-bold text-sm mb-3">Leyenda</h5>
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full bg-red-500 border border-red-700 opacity-80" />
                  <span className="text-xs">Potencial Alto</span>
                </div>
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full bg-orange-400 border border-orange-600 opacity-80" />
                  <span className="text-xs">Potencial Medio</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500 border border-blue-700 relative">
                    <div className="absolute w-full h-full bg-blue-500 rounded-full animate-ping opacity-50" />
                  </div>
                  <span className="text-xs">Tu selección</span>
                </div>
              </div>
            </div>
          </div>

          {/* PANEL DE ACCION */}
          <div className="w-full xl:w-1/3 flex flex-col gap-6">
            <div className="bg-[#1a1d27] rounded-[2rem] p-8 border border-white/5 relative overflow-hidden">
              <h3 className="text-sm font-mono tracking-widest text-gray-400 mb-6 border-b border-white/5 pb-4">
                PANEL DE INFERENCIA
              </h3>

              {inputMode === "zone" ? (
                <div className="mb-8 p-4 bg-[#0b0c10] rounded-xl border border-white/5">
                  <label className="block text-[10px] tracking-widest font-mono text-gray-600 mb-2">PUNTO DE INTERÉS</label>
                  <select
                    onChange={(e) => {
                      const [lat, lng] = e.target.value.split(",");
                      setCoords({ lat: parseFloat(lat), lng: parseFloat(lng) });
                    }}
                    className="w-full bg-transparent text-white font-medium focus:outline-none text-base p-2"
                  >
                    {zonasGeotermicas.map((z) => (
                      <option key={z.nombre} className="bg-[#1a1d27]" value={`${z.lat},${z.lon}`}>
                        {z.nombre}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div className="bg-[#0b0c10] rounded-xl p-4 border border-white/5">
                    <label className="block text-[10px] tracking-widest font-mono text-gray-600 mb-1">LATITUD</label>
                    <input
                      type="number"
                      step="any"
                      value={coords.lat}
                      onChange={(e) => inputMode === "coords" && setCoords({ ...coords, lat: parseFloat(e.target.value) || 0 })}
                      disabled={inputMode === "map"}
                      className="w-full bg-transparent text-xl text-white font-medium focus:outline-none focus:text-cyan-400 disabled:opacity-80"
                    />
                  </div>
                  <div className="bg-[#0b0c10] rounded-xl p-4 border border-white/5">
                    <label className="block text-[10px] tracking-widest font-mono text-gray-600 mb-1">LONGITUD</label>
                    <input
                      type="number"
                      step="any"
                      value={coords.lng}
                      onChange={(e) => inputMode === "coords" && setCoords({ ...coords, lng: parseFloat(e.target.value) || 0 })}
                      disabled={inputMode === "map"}
                      className="w-full bg-transparent text-xl text-white font-medium focus:outline-none focus:text-cyan-400 disabled:opacity-80"
                    />
                  </div>
                </div>
              )}

              <button
                onClick={handlePredict}
                disabled={loading}
                className="w-full bg-white text-black py-4 rounded-xl font-bold transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.3)] flex justify-center items-center gap-2 active:scale-[0.98]"
              >
                {loading ? (
                  <span className="animate-spin border-2 border-black/30 border-t-black w-5 h-5 rounded-full" />
                ) : (
                  <>
                    <Database size={18} /> COMPUTAR MODELO
                  </>
                )}
              </button>
            </div>

            {/* Resultado */}
            <div className="flex-grow bg-[#1a1d27] rounded-[2rem] p-6 border border-white/5 flex justify-center items-center text-center relative overflow-hidden min-h-[280px]">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-red-900/10 via-transparent to-transparent opacity-50 mix-blend-screen pointer-events-none" />
              {prediction !== null && !loading && prediction !== "ERROR" ? (
                <div className="relative z-10 w-full">
                  <p className="text-[10px] font-mono text-gray-500 tracking-[0.3em] uppercase mb-4">Potencial Calculado</p>
                  <div
                    className={`text-6xl md:text-7xl font-black text-transparent bg-clip-text drop-shadow-[0_0_15px_rgba(255,255,255,0.1)] mb-6 ${
                      parseFloat(prediction) > 70
                        ? "bg-gradient-to-br from-white via-orange-200 to-red-500"
                        : parseFloat(prediction) > 30
                        ? "bg-gradient-to-br from-white via-yellow-200 to-orange-500"
                        : "bg-gradient-to-br from-white via-gray-200 to-gray-500"
                    }`}
                  >
                    {prediction}%
                  </div>
                  <div className="flex flex-col items-center gap-2">
                    {parseFloat(prediction) >= 70 ? (
                      <div className="inline-flex items-center gap-2 bg-red-500/10 border border-red-500/30 px-4 py-2 rounded-full">
                        <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
                        <span className="text-xs font-bold text-red-500">POTENCIAL ALTO</span>
                      </div>
                    ) : parseFloat(prediction) >= 30 ? (
                      <div className="inline-flex items-center gap-2 bg-orange-500/10 border border-orange-500/30 px-4 py-2 rounded-full">
                        <span className="w-2 h-2 rounded-full bg-orange-500" />
                        <span className="text-xs font-bold text-orange-500">POTENCIAL MEDIO</span>
                      </div>
                    ) : (
                      <div className="inline-flex items-center gap-2 bg-gray-500/10 border border-gray-500/30 px-4 py-2 rounded-full">
                        <span className="w-2 h-2 rounded-full bg-gray-400" />
                        <span className="text-xs font-bold text-gray-400">POTENCIAL BAJO</span>
                      </div>
                    )}
                    {result?.zona_cercana && (
                      <p className="text-xs text-gray-400 mt-2 font-light">
                        Zona ref: <span className="text-white font-medium">{result.zona_cercana}</span>
                        {result.distancia_km != null && (
                          <span className="text-gray-600 ml-1">({result.distancia_km} km)</span>
                        )}
                      </p>
                    )}
                    {result?.metodo && (
                      <p className="text-[10px] text-gray-500 font-mono mt-1">
                        Método: <span className={result.metodo === "cnn" ? "text-green-400" : "text-orange-400"}>{result.metodo.toUpperCase()}</span>
                        {result.modelo && <span className="text-gray-600"> · {result.modelo}</span>}
                      </p>
                    )}
                  </div>
                </div>
              ) : prediction === "ERROR" ? (
                <div className="text-red-500 relative z-10">
                  <Activity size={40} className="mx-auto mb-4" />
                  <span className="text-xs font-mono block">{result?.error || "Error de conexión con API"}</span>
                </div>
              ) : loading ? (
                <div className="text-cyan-500 relative z-10">
                  <Cpu size={40} className="mx-auto mb-4 animate-bounce" />
                  <span className="text-[10px] font-mono tracking-widest animate-pulse">EVALUANDO TENSOR_</span>
                </div>
              ) : (
                <div className="opacity-20">
                  <Database size={40} strokeWidth={1} className="mx-auto mb-4" />
                  <span className="text-[10px] uppercase font-mono tracking-widest">Esperando Ubicación</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── Detalles Satélite + Bandas ── */}
      {result && !result.error && result.satelite && (
        <section className="w-full max-w-6xl mb-10">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Info Satélite */}
            <div className="bg-[#12141c]/70 backdrop-blur-md rounded-3xl border border-white/5 p-6">
              <h3 className="text-sm font-mono tracking-widest text-gray-400 mb-5 flex items-center gap-2 border-b border-white/5 pb-3">
                <Satellite size={16} className="text-cyan-400" /> INFORMACIÓN DEL SATÉLITE
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                {[
                  { label: "Dataset", val: result.satelite.dataset },
                  { label: "Sensor", val: result.satelite.sensor },
                  { label: "Resolución", val: `${result.satelite.resolucion_m} m/px` },
                  { label: "Buffer", val: `${result.satelite.buffer_m} m` },
                  { label: "CRS", val: result.satelite.crs },
                  { label: "Forma Imagen", val: result.satelite.imagen_shape?.join(" × ") },
                  { label: "Tamaño", val: `${result.satelite.archivo_kb} KB` },
                ].map(({ label, val }) => (
                  <div key={label} className="bg-black/20 rounded-xl px-4 py-3 border border-white/5">
                    <p className="text-[10px] text-gray-500 font-mono tracking-widest mb-1">{label.toUpperCase()}</p>
                    <p className="text-white font-mono text-xs truncate" title={val}>{val}</p>
                  </div>
                ))}
                {result.tiempos && (
                  <div className="bg-black/20 rounded-xl px-4 py-3 border border-white/5">
                    <p className="text-[10px] text-gray-500 font-mono tracking-widest mb-1">TIEMPOS</p>
                    <p className="text-cyan-400 font-mono text-xs">
                      ↓{result.tiempos.descarga}s · ⚡{result.tiempos.prediccion}s · Σ{result.tiempos.total}s
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Info Bandas */}
            <div className="bg-[#12141c]/70 backdrop-blur-md rounded-3xl border border-white/5 p-6">
              <h3 className="text-sm font-mono tracking-widest text-gray-400 mb-5 flex items-center gap-2 border-b border-white/5 pb-3">
                <Layers size={16} className="text-orange-400" /> ESTADÍSTICAS POR BANDA
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-gray-500 text-[10px] tracking-widest">
                      <th className="text-left py-2 px-2">BANDA</th>
                      <th className="text-right py-2 px-2">MIN</th>
                      <th className="text-right py-2 px-2">MAX</th>
                      <th className="text-right py-2 px-2">MEDIA</th>
                      <th className="text-right py-2 px-2">STD</th>
                      <th className="text-right py-2 px-2">NODATA</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.bandas?.map((b, i) => (
                      <tr key={i} className="border-t border-white/5 hover:bg-white/5 transition-colors">
                        <td className="py-2 px-2 text-left">
                          <span className={`inline-block w-2 h-2 rounded-full mr-2 ${
                            b.nombre.includes("temperature") ? "bg-red-400"
                              : b.nombre.includes("ndvi") ? "bg-green-400"
                              : "bg-cyan-400"
                          }`} />
                          <span className="text-gray-300">{b.nombre.replace("emissivity_", "emis_")}</span>
                        </td>
                        <td className="py-2 px-2 text-right text-gray-400">{b.min}</td>
                        <td className="py-2 px-2 text-right text-gray-400">{b.max}</td>
                        <td className="py-2 px-2 text-right text-white">{b.mean}</td>
                        <td className="py-2 px-2 text-right text-gray-400">{b.std}</td>
                        <td className="py-2 px-2 text-right">
                          <span className={b.nodata_pct > 50 ? "text-red-400" : b.nodata_pct > 0 ? "text-orange-400" : "text-green-400"}>
                            {b.nodata_pct}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {result.bandas && (
                <p className="text-[10px] text-gray-600 mt-3 font-mono">
                  {result.bandas.length} bandas · valores antes de normalización z-score
                </p>
              )}
            </div>
          </div>
        </section>
      )}

      {/* ── Mapa de Calor — Historial de Predicciones ── */}
      <section className="w-full max-w-6xl mb-16">
        <div className="bg-[#12141c]/60 backdrop-blur-md rounded-3xl border border-white/5 p-6 md:p-8">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-white/5 pb-4">
            <h3 className="text-sm font-mono tracking-widest text-gray-400 flex items-center gap-2">
              <ThermometerSun size={16} className="text-red-400" /> MAPA DE CALOR — ÚLTIMAS {history.length} PREDICCIONES
            </h3>
            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="flex items-center gap-1.5 text-[10px] text-gray-500 hover:text-red-400 transition-colors font-mono uppercase tracking-widest"
              >
                <Trash2 size={12} /> Limpiar historial
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div className="text-center py-16 text-gray-600">
              <ThermometerSun size={40} strokeWidth={1} className="mx-auto mb-4 opacity-30" />
              <p className="text-sm font-mono">Aún no hay predicciones. Ejecuta una para que aparezca aquí.</p>
            </div>
          ) : (
            <div className="grid lg:grid-cols-5 gap-6">
              {/* Mini-mapa con heatmarkers */}
              <div className="lg:col-span-3 h-[360px] rounded-2xl overflow-hidden border border-white/5">
                <MapContainer
                  center={[4.5709, -74.2973]}
                  zoom={5.5}
                  style={{ height: "100%", width: "100%" }}
                  scrollWheelZoom={false}
                >
                  <TileLayer
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    attribution="&copy; CARTO"
                  />
                  {history.map((h, i) => (
                    <CircleMarker
                      key={i}
                      center={[h.lat, h.lon]}
                      pathOptions={{
                        color: heatColor(h.porcentaje),
                        fillColor: heatColor(h.porcentaje),
                        fillOpacity: 0.7 - i * 0.05,
                        weight: i === 0 ? 3 : 1,
                      }}
                      radius={i === 0 ? 14 : 10}
                    >
                      <Popup>
                        <strong>{h.porcentaje}%</strong> · {h.zona}<br />
                        <span className="text-[10px]">{new Date(h.timestamp).toLocaleString()}</span>
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>
              </div>

              {/* Tabla historial */}
              <div className="lg:col-span-2 overflow-y-auto max-h-[360px]">
                <table className="w-full text-xs font-mono">
                  <thead className="sticky top-0 bg-[#12141c]">
                    <tr className="text-gray-500 text-[10px] tracking-widest">
                      <th className="text-left py-2 px-2">#</th>
                      <th className="text-left py-2 px-2">COORDS</th>
                      <th className="text-right py-2 px-2">%</th>
                      <th className="text-left py-2 px-2">ZONA</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((h, i) => (
                      <tr key={i} className="border-t border-white/5 hover:bg-white/5 transition-colors">
                        <td className="py-2 px-2 text-gray-600">{i + 1}</td>
                        <td className="py-2 px-2 text-gray-400">
                          {h.lat.toFixed(3)}, {h.lon.toFixed(3)}
                        </td>
                        <td className="py-2 px-2 text-right font-bold" style={{ color: heatColor(h.porcentaje) }}>
                          {h.porcentaje}%
                        </td>
                        <td className="py-2 px-2 text-gray-300 truncate max-w-[120px]" title={h.zona}>
                          {h.zona}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* Leyenda */}
                <div className="mt-4 flex items-center gap-3 px-2 text-[10px] text-gray-500 font-mono">
                  <span>Leyenda:</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500" /> ≥70%</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-orange-500" /> ≥50%</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-500" /> ≥30%</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-500" /> &lt;30%</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
