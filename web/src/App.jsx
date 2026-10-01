import { BrowserRouter, Routes, Route } from "react-router-dom";
import AppLayout from "./layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import DatasetNew from "./pages/DatasetNew";
import DatasetDetail from "./pages/DatasetDetail";
import Models from "./pages/Models";
import ModelNew from "./pages/ModelNew";
import ModelDetail from "./pages/ModelDetail";
import Jobs from "./pages/Jobs";
import CreateJob from "./pages/CreateJob";
import JobDetail from "./pages/JobDetail";
import Checkpoints from "./pages/Checkpoints";
import Deployments from "./pages/Deployments";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/datasets/new" element={<DatasetNew />} />
          <Route path="/datasets/:id" element={<DatasetDetail />} />
          <Route path="/models" element={<Models />} />
          <Route path="/models/new" element={<ModelNew />} />
          <Route path="/models/:id" element={<ModelDetail />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/jobs/new" element={<CreateJob />} />
          <Route path="/jobs/:jobId" element={<JobDetail />} />
          <Route path="/checkpoints" element={<Checkpoints />} />
          <Route path="/deployments" element={<Deployments />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
