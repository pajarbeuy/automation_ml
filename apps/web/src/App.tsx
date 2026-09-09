import { BrowserRouter, Routes, Route, Navigate, useParams } from 'react-router-dom';
import { DashboardPage } from './pages/DashboardPage';
import { ProjectPage } from './pages/ProjectPage';
import { UploadPage } from './pages/UploadPage';
import { EDAPage } from './pages/EDAPage';
import { TreatmentPage } from './pages/TreatmentPage';
import { PipelinePage } from './pages/PipelinePage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { ResultPage } from './pages/ResultPage';
import { Sidebar } from './components/Sidebar';
import { AIAssistantWidget } from './components/AIAssistantWidget';
import './index.css';

function ProjectLayout() {
  const { projectId } = useParams<{ projectId: string }>();
  return (
    <>
      <Sidebar />
      <div className="main-content">
        <Routes>
          <Route index element={<Navigate to="upload" replace />} />
          <Route path="overview" element={<ProjectPage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="eda" element={<EDAPage />} />
          <Route path="treatment" element={<TreatmentPage />} />
          <Route path="pipeline" element={<PipelinePage />} />
          <Route path="experiments" element={<ExperimentsPage />} />
          <Route path="result" element={<ResultPage />} />
        </Routes>
      </div>
      <AIAssistantWidget projectId={projectId} />
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout-main" style={{ flexDirection: 'row' }}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/projects/:projectId/*" element={<ProjectLayout />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
