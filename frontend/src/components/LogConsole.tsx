import { useEffect, useRef } from 'react'

export interface LogEntry {
  ts: string
  type: 'info' | 'score' | 'error' | 'warn' | 'done'
  body: string
}

export default function LogConsole({ logs }: { logs: LogEntry[] }) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  return (
    <div className="log-console">
      {logs.length === 0 ? (
        <span className="log-empty">
          › Waiting for pipeline to start…
        </span>
      ) : (
        logs.map((l, i) => (
          <div key={i} className={`log-line log-${l.type}`}>
            <span className="log-ts">[{l.ts}]</span>
            <span className="log-body">{l.body}</span>
          </div>
        ))
      )}
      <div ref={bottomRef} />
    </div>
  )
}
