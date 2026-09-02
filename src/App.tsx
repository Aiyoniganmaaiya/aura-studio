import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import GeneratePage from "./pages/GeneratePage";
import ModelsPage from "./pages/ModelsPage";
import HistoryPage from "./pages/HistoryPage";
import SettingsPage from "./pages/SettingsPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<GeneratePage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="settings" element={<SettingsPage />} />
        {/* Unknown paths fall back to the generator instead of a blank page. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

export default App;
