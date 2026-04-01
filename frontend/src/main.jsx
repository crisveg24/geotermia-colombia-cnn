import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import PrediccionPage from "./pages/PrediccionPage";
import MetricasPage from "./pages/MetricasPage";
import ArquitecturaPage from "./pages/ArquitecturaPage";
import ProyectoPage from "./pages/ProyectoPage";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="prediccion" element={<PrediccionPage />} />
          <Route path="metricas" element={<MetricasPage />} />
          <Route path="arquitectura" element={<ArquitecturaPage />} />
          <Route path="proyecto" element={<ProyectoPage />} />
          <Route path="*" element={<HomePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);