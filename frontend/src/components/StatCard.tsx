interface StatCardProps {
  label: string
  value: string | number
  meta: string
  sub: string
  color: 'blue' | 'red' | 'yellow' | 'green'
  metaVariant?: 'positive' | 'warning' | 'danger' | 'success'
  icon: React.ReactNode
}

export default function StatCard({ label, value, meta, sub, color, metaVariant = 'positive', icon }: StatCardProps) {
  return (
    <div className="stat-card">
      <div className="stat-card-header">
        <div className="stat-card-label">{label}</div>
        <div className={`stat-card-icon ${color}`}>{icon}</div>
      </div>
      <div className="stat-card-value">{value}</div>
      <div className={`stat-card-meta ${metaVariant}`}>{meta}</div>
      <div className="stat-card-sub">{sub}</div>
    </div>
  )
}
