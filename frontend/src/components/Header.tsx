interface HeaderProps {
  pipelineStatus: 'idle' | 'running' | 'done' | 'error'
  searchValue: string
  onSearchChange: (v: string) => void
}

export default function Header({ pipelineStatus, searchValue, onSearchChange }: HeaderProps) {
  const statusLabel =
    pipelineStatus === 'running'
      ? 'Pipeline: running…'
      : pipelineStatus === 'done'
      ? 'Pipeline: done'
      : pipelineStatus === 'error'
      ? 'Pipeline: error'
      : 'Pipeline: idle'

  return (
    <header className="app-header">
      <div className="header-search">
        <svg className="search-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <circle cx="11" cy="11" r="8"/>
          <path d="M21 21l-4.35-4.35"/>
        </svg>
        <input
          type="text"
          placeholder="Search customer ID…"
          value={searchValue}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </div>

      <div className="header-spacer" />

      <div className="pipeline-pill">
        <span className={`pipeline-dot ${pipelineStatus}`} />
        {statusLabel}
      </div>

      <div className="header-avatar">AN</div>
    </header>
  )
}
