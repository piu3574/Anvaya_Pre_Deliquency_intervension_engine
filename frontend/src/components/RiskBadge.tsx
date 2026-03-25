type Band = 'GREEN' | 'YELLOW' | 'RED' | string

export default function RiskBadge({ band }: { band: Band }) {
  return <span className={`risk-badge ${band}`}>{band}</span>
}
