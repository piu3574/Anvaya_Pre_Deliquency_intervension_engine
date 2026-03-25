import { useRef, useCallback } from 'react'

export interface SSEMessage {
  type: 'log' | 'score' | 'done' | 'error'
  message?: string
  data?: Record<string, unknown>
}

export function useSSE() {
  const sourceRef = useRef<EventSource | null>(null)

  const start = useCallback(
    (
      url: string,
      onMessage: (msg: SSEMessage) => void,
      onDone: () => void,
    ) => {
      if (sourceRef.current) {
        sourceRef.current.close()
      }

      const source = new EventSource(url)
      sourceRef.current = source

      source.onmessage = (e) => {
        try {
          const parsed: SSEMessage = JSON.parse(e.data)
          onMessage(parsed)
          if (parsed.type === 'done') {
            source.close()
            onDone()
          }
        } catch {
          /* ignore malformed frames */
        }
      }

      source.onerror = () => {
        onMessage({ type: 'error', message: 'Connection lost.' })
        source.close()
        onDone()
      }
    },
    [],
  )

  const stop = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
  }, [])

  return { start, stop }
}
