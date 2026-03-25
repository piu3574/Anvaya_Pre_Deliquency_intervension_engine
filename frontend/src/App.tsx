import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Dashboard from './pages/Dashboard'
import CustomerList from './pages/CustomerList'
import PipelineMonitor from './pages/PipelineMonitor'
import ModelHealth from './pages/ModelHealth'
import './index.css'

export default function App() {
  const [search, setSearch] = useState('')
  // pipelineStatus is lifted here so Header can show it from PipelineMonitor context
  // For simplicity we keep it as idle here; PipelineMonitor manages its own state
  const pipelineStatus: 'idle' | 'running' | 'done' | 'error' = 'idle'

  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <div className="main-area">
          <Header
            pipelineStatus={pipelineStatus}
            searchValue={search}
            onSearchChange={setSearch}
          />
          <main className="page-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/customers" element={<CustomerList />} />
              <Route path="/pipeline" element={<PipelineMonitor />} />
              <Route path="/model-health" element={<ModelHealth />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  )
}
