import { BrowserRouter, Routes, Route, Navigate, useParams } from 'react-router-dom';
import { DashboardPage } from './pages/DashboardPage';
import { ProjectPage } from './pages/ProjectPage';
import { UploadPage } from './pages/UploadPage';
import { EDAPage } from './pages/EDAPage';
import { TreatmentPage } from './pages/TreatmentPage';
import { PipelinePage } from './pages/PipelinePage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { ResultPage } from './pages/ResultPage';
import { DriftMonitoringPage } from './pages/DriftMonitoringPage';
import { Sidebar } from './components/Sidebar';
import { AIAssistantWidget } from './components/AIAssistantWidget';
import { ErrorBoundary } from './components/ErrorBoundary';
import './index.css';

function ProjectLayout() {
  const { projectId } = useParams<{ projectId: string }>();
  return (
    <>
      <Sidebar />
      <div className="main-content">
        <ErrorBoundary>
          <Routes>
            <Route index element={<Navigate to="upload" replace />} />
            <Route path="overview" element={<ProjectPage />} />
            <Route path="upload" element={<UploadPage />} />
            <Route path="eda" element={<EDAPage />} />
            <Route path="treatment" element={<TreatmentPage />} />
            <Route path="pipeline" element={<PipelinePage />} />
            <Route path="experiments" element={<ExperimentsPage />} />
            <Route path="result" element={<ResultPage />} />
            <Route path="drift" element={<DriftMonitoringPage />} />
          </Routes>
        </ErrorBoundary>
      </div>
      <AIAssistantWidget projectId={projectId} />
    </>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <div className="layout-main">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/projects/:projectId/*" element={<ProjectLayout />} />
          </Routes>
        </div>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
