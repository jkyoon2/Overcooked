import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import GameView from '../components/GameView'
import type {
  PhaseTwoSegmentSelection,
  PhaseTwoStartPayload,
  PhaseTwoTrialSelection,
} from '../lib/websocket'

type PhaseTwoScreenProps = {
  data: PhaseTwoStartPayload
  submitting: boolean
  onSubmit: (trials: PhaseTwoTrialSelection[]) => void
}

type DragRange = {
  startFrame: number
  endFrame: number
}

type ResizeState = {
  segmentId: string
  edge: 'start' | 'end'
}

export default function PhaseTwoScreen({
  data,
  submitting,
  onSubmit,
}: PhaseTwoScreenProps) {
  const [trialIndex, setTrialIndex] = useState(0)
  const [frameIndex, setFrameIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [segmentEndFrame, setSegmentEndFrame] = useState<number | null>(null)
  const [dragRange, setDragRange] = useState<DragRange | null>(null)
  const [segmentsByTrial, setSegmentsByTrial] = useState<
    Record<number, PhaseTwoSegmentSelection[]>
  >({})
  const [resizeState, setResizeState] = useState<ResizeState | null>(null)
  const timelineRef = useRef<HTMLDivElement>(null)
  const dragStartRef = useRef<number | null>(null)
  const segmentSequenceRef = useRef(0)

  const trial = data.trials[trialIndex]
  const frames = trial?.frames ?? []
  const maxFrame = Math.max(frames.length - 1, 0)
  const currentFrame = frames[Math.min(frameIndex, maxFrame)]
  const currentSegments = trial ? segmentsByTrial[trial.trial_id] ?? [] : []
  const playbackEnd = segmentEndFrame ?? maxFrame

  useEffect(() => {
    setFrameIndex(0)
    setSegmentEndFrame(null)
    setDragRange(null)
    setIsPlaying(true)
  }, [trialIndex])

  useEffect(() => {
    if (!isPlaying || frames.length === 0) return
    const intervalId = window.setInterval(() => {
      setFrameIndex((previous) => Math.min(previous + 1, playbackEnd))
    }, Math.max(16, data.frame_duration_ms / playbackRate))
    return () => window.clearInterval(intervalId)
  }, [
    data.frame_duration_ms,
    frames.length,
    isPlaying,
    playbackEnd,
    playbackRate,
  ])

  useEffect(() => {
    if (isPlaying && frameIndex >= playbackEnd) {
      setIsPlaying(false)
      setSegmentEndFrame(null)
    }
  }, [frameIndex, isPlaying, playbackEnd])

  if (!trial || !currentFrame) {
    return (
      <main style={emptyPageStyle}>
        <h1>Phase 2</h1>
        <p>No replay frames were recorded.</p>
      </main>
    )
  }

  const seek = (targetFrame: number) => {
    setFrameIndex(clamp(targetFrame, 0, maxFrame))
    setIsPlaying(false)
    setSegmentEndFrame(null)
  }

  const togglePlay = () => {
    setSegmentEndFrame(null)
    if (frameIndex >= maxFrame) {
      setFrameIndex(0)
      setIsPlaying(true)
      return
    }
    setIsPlaying((playing) => !playing)
  }

  const replaySegment = (segment: PhaseTwoSegmentSelection) => {
    setFrameIndex(segment.start_frame)
    setSegmentEndFrame(segment.end_frame)
    setIsPlaying(true)
  }

  const frameFromPointer = (clientX: number): number => {
    const timeline = timelineRef.current
    if (!timeline || maxFrame === 0) return 0
    const bounds = timeline.getBoundingClientRect()
    const progress = clamp((clientX - bounds.left) / bounds.width, 0, 1)
    return Math.round(progress * maxFrame)
  }

  const beginDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (resizeState !== null) return
    event.currentTarget.setPointerCapture(event.pointerId)
    const targetFrame = frameFromPointer(event.clientX)
    dragStartRef.current = targetFrame
    setDragRange({ startFrame: targetFrame, endFrame: targetFrame })
    seek(targetFrame)
  }

  const updateDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const startFrame = dragStartRef.current
    if (startFrame === null) return
    const endFrame = frameFromPointer(event.clientX)
    setDragRange({ startFrame, endFrame })
    setFrameIndex(endFrame)
  }

  const finishDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const startFrame = dragStartRef.current
    if (startFrame === null) return
    const pointerFrame = frameFromPointer(event.clientX)
    const rangeStart = Math.min(startFrame, pointerFrame)
    const rangeEnd = Math.max(startFrame, pointerFrame)
    dragStartRef.current = null
    setDragRange(null)

    segmentSequenceRef.current += 1
    const segment: PhaseTwoSegmentSelection = {
      segment_id: `${trial.trial_id}-${Date.now()}-${segmentSequenceRef.current}`,
      start_frame: rangeStart,
      end_frame: rangeEnd,
      created_at_ms: Date.now(),
    }
    setSegmentsByTrial((current) => ({
      ...current,
      [trial.trial_id]: [...(current[trial.trial_id] ?? []), segment],
    }))
  }

  const cancelDrag = () => {
    dragStartRef.current = null
    setDragRange(null)
  }

  const beginResize = (
    segmentId: string,
    edge: 'start' | 'end',
    event: ReactPointerEvent<HTMLDivElement>,
  ) => {
    event.stopPropagation()
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    setResizeState({ segmentId, edge })
    setIsPlaying(false)
    setSegmentEndFrame(null)
  }

  const updateResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (resizeState === null) return
    event.stopPropagation()
    const pointerFrame = frameFromPointer(event.clientX)
    setSegmentsByTrial((current) => {
      const trialSegments = current[trial.trial_id] ?? []
      const next = trialSegments.map((segment) => {
        if (segment.segment_id !== resizeState.segmentId) return segment
        const nextStart =
          resizeState.edge === 'start' ? pointerFrame : segment.start_frame
        const nextEnd =
          resizeState.edge === 'end' ? pointerFrame : segment.end_frame
        return {
          ...segment,
          start_frame: Math.min(nextStart, nextEnd),
          end_frame: Math.max(nextStart, nextEnd),
        }
      })
      return { ...current, [trial.trial_id]: next }
    })
    setFrameIndex(clamp(pointerFrame, 0, maxFrame))
  }

  const finishResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (resizeState === null) return
    event.stopPropagation()
    setResizeState(null)
  }

  const deleteSegment = (segmentId: string) => {
    setSegmentsByTrial((current) => ({
      ...current,
      [trial.trial_id]: (current[trial.trial_id] ?? []).filter(
        (segment) => segment.segment_id !== segmentId,
      ),
    }))
  }

  const submitPhaseTwo = () => {
    onSubmit(
      data.trials.map((replayTrial) => ({
        trial_id: replayTrial.trial_id,
        segments: segmentsByTrial[replayTrial.trial_id] ?? [],
      })),
    )
  }

  const normalizedDrag = dragRange === null
    ? null
    : {
        start_frame: Math.min(dragRange.startFrame, dragRange.endFrame),
        end_frame: Math.max(dragRange.startFrame, dragRange.endFrame),
      }

  return (
    <main style={pageStyle}>
      <header style={headerStyle}>
        <div>
          <strong style={{ fontSize: '1.05rem' }}>Neurocontroller</strong>
          <span style={phaseLabelStyle}>Phase 2, Replay Trial {trial.trial_id}</span>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {data.trials.map((replayTrial, index) => {
            const count = segmentsByTrial[replayTrial.trial_id]?.length ?? 0
            return (
              <button
                key={replayTrial.trial_id}
                type="button"
                onClick={() => setTrialIndex(index)}
                style={{
                  ...trialTabStyle,
                  borderColor: index === trialIndex ? '#2563eb' : '#cbd5e1',
                  background: index === trialIndex ? '#eff6ff' : '#ffffff',
                  color: index === trialIndex ? '#1d4ed8' : '#475569',
                }}
              >
                Trial {replayTrial.trial_id}
                {count > 0 && <span style={countBadgeStyle}>{count}</span>}
              </button>
            )
          })}
        </div>
      </header>

      <div style={contentStyle}>
        <section style={viewerColumnStyle}>
          <div style={introStyle}>
            <div>
              <h1 style={{ margin: 0, fontSize: '1.55rem' }}>
                Mark behavior that did not align with your strategy
              </h1>
              <p style={{ margin: '0.4rem 0 0', color: '#475569' }}>
                Replay the trial, then drag across the timeline to select each
                misaligned interval. Drag the left or right handle of a saved
                segment to adjust either endpoint.
              </p>
            </div>
            <div style={timeBadgeStyle}>
              {formatTime(frameIndex, data.frame_duration_ms)} /{' '}
              {formatTime(maxFrame, data.frame_duration_ms)}
            </div>
          </div>

          <div style={gamePanelStyle} onClick={togglePlay}>
            <GameView
              state={currentFrame}
              playerHat={data.player_hat}
              aiHat={data.ai_hat}
            />
            {!isPlaying && (
              <div style={pausedOverlayStyle}>
                <span>{frameIndex >= maxFrame ? 'Replay finished' : 'Paused'}</span>
              </div>
            )}
          </div>

          <div style={controlsPanelStyle}>
            <div
              ref={timelineRef}
              role="slider"
              aria-label="Drag to select a misaligned replay interval"
              aria-valuemin={0}
              aria-valuemax={maxFrame}
              aria-valuenow={frameIndex}
              tabIndex={0}
              onPointerDown={beginDrag}
              onPointerMove={updateDrag}
              onPointerUp={finishDrag}
              onPointerCancel={cancelDrag}
              style={timelineStyle}
            >
              <div style={timelineTrackStyle} />
              {currentSegments.map((segment) => {
                const position = rangePosition(
                  segment.start_frame,
                  segment.end_frame,
                  maxFrame,
                )
                const isResizing = resizeState?.segmentId === segment.segment_id
                return (
                  <div
                    key={segment.segment_id}
                    style={{
                      ...selectionRangeStyle,
                      ...position,
                      background: isResizing
                        ? 'rgba(37, 99, 235, 0.42)'
                        : 'rgba(37, 99, 235, 0.28)',
                      borderColor: '#2563eb',
                      pointerEvents: 'none',
                    }}
                  >
                    <div
                      role="slider"
                      aria-label={`Adjust start of segment from frame ${segment.start_frame}`}
                      aria-valuemin={0}
                      aria-valuemax={maxFrame}
                      aria-valuenow={segment.start_frame}
                      onPointerDown={(event) => beginResize(segment.segment_id, 'start', event)}
                      onPointerMove={updateResize}
                      onPointerUp={finishResize}
                      onPointerCancel={finishResize}
                      style={{
                        ...handleStyle,
                        left: 0,
                        background:
                          isResizing && resizeState?.edge === 'start'
                            ? '#1d4ed8'
                            : '#2563eb',
                      }}
                    />
                    <div
                      role="slider"
                      aria-label={`Adjust end of segment to frame ${segment.end_frame}`}
                      aria-valuemin={0}
                      aria-valuemax={maxFrame}
                      aria-valuenow={segment.end_frame}
                      onPointerDown={(event) => beginResize(segment.segment_id, 'end', event)}
                      onPointerMove={updateResize}
                      onPointerUp={finishResize}
                      onPointerCancel={finishResize}
                      style={{
                        ...handleStyle,
                        right: 0,
                        background:
                          isResizing && resizeState?.edge === 'end'
                            ? '#1d4ed8'
                            : '#2563eb',
                      }}
                    />
                  </div>
                )
              })}
              {normalizedDrag !== null && (
                <div
                  style={{
                    ...selectionRangeStyle,
                    ...rangePosition(
                      normalizedDrag.start_frame,
                      normalizedDrag.end_frame,
                      maxFrame,
                    ),
                    background: 'rgba(239, 68, 68, 0.24)',
                    borderColor: '#ef4444',
                    pointerEvents: 'none',
                  }}
                />
              )}
              <div
                style={{
                  ...playheadStyle,
                  left: `${framePercent(frameIndex, maxFrame)}%`,
                }}
              />
            </div>

            <div style={controlsRowStyle}>
              <span style={frameLabelStyle}>Frame {frameIndex} / {maxFrame}</span>
              <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
                <ControlButton label="First frame" onClick={() => seek(0)}>⏮</ControlButton>
                <ControlButton label="Previous frame" onClick={() => seek(frameIndex - 1)}>
                  ◀
                </ControlButton>
                <ControlButton label={isPlaying ? 'Pause' : 'Play'} onClick={togglePlay} primary>
                  {isPlaying ? 'Ⅱ' : '▶'}
                </ControlButton>
                <ControlButton label="Next frame" onClick={() => seek(frameIndex + 1)}>
                  ▶
                </ControlButton>
                <ControlButton label="Last frame" onClick={() => seek(maxFrame)}>⏭</ControlButton>
              </div>
              <label style={speedLabelStyle}>
                Speed
                <select
                  value={playbackRate}
                  onChange={(event) => setPlaybackRate(Number(event.target.value))}
                  style={speedSelectStyle}
                >
                  <option value={0.5}>0.5x</option>
                  <option value={1}>1x</option>
                  <option value={1.5}>1.5x</option>
                  <option value={2}>2x</option>
                </select>
              </label>
            </div>
            <p style={dragHintStyle}>
              Drag across an empty area of the timeline to save a new segment.
              Drag a segment&apos;s left or right edge to fine-tune its endpoints.
            </p>
          </div>
        </section>

        <aside style={sidePanelStyle}>
          <div>
            <p style={panelEyebrowStyle}>Trial {trial.trial_id}</p>
            <h2 style={{ margin: 0 }}>Misaligned segments</h2>
            <p style={{ margin: '0.5rem 0 0', color: '#475569', lineHeight: 1.5 }}>
              Selected intervals appear here. Replay or remove them before continuing.
            </p>
          </div>

          <div style={segmentListStyle}>
            {currentSegments.length === 0 ? (
              <div style={emptySegmentsStyle}>
                <strong>No segments selected</strong>
                <span>Drag across the timeline below the replay.</span>
              </div>
            ) : (
              currentSegments.map((segment, index) => (
                <article
                  key={segment.segment_id}
                  onClick={() => seek(segment.start_frame)}
                  style={segmentCardStyle}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
                    <div>
                      <strong>Misaligned segment {index + 1}</strong>
                      <p style={segmentTimeStyle}>
                        {formatTime(segment.start_frame, data.frame_duration_ms)}
                        {' - '}
                        {formatTime(segment.end_frame, data.frame_duration_ms)}
                      </p>
                    </div>
                    <button
                      type="button"
                      aria-label={`Delete segment ${index + 1}`}
                      onClick={(event) => {
                        event.stopPropagation()
                        deleteSegment(segment.segment_id)
                      }}
                      style={deleteButtonStyle}
                    >
                      ×
                    </button>
                  </div>
                  <div style={segmentMetaStyle}>
                    Frames {segment.start_frame}-{segment.end_frame}
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation()
                        replaySegment(segment)
                      }}
                      style={replayButtonStyle}
                    >
                      ▶ Replay segment
                    </button>
                  </div>
                </article>
              ))
            )}
          </div>

          <div style={panelActionsStyle}>
            {trialIndex < data.trials.length - 1 ? (
              <button
                type="button"
                onClick={() => setTrialIndex((index) => index + 1)}
                style={nextButtonStyle}
              >
                Next replay
              </button>
            ) : (
              <button
                type="button"
                disabled={submitting}
                onClick={submitPhaseTwo}
                style={{
                  ...nextButtonStyle,
                  opacity: submitting ? 0.65 : 1,
                  cursor: submitting ? 'wait' : 'pointer',
                }}
              >
                {submitting ? 'Saving...' : 'Finish Phase 2'}
              </button>
            )}
          </div>
        </aside>
      </div>
    </main>
  )
}

type ControlButtonProps = {
  children: string
  label: string
  onClick: () => void
  primary?: boolean
}

function ControlButton({
  children,
  label,
  onClick,
  primary = false,
}: ControlButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={(event) => {
        event.stopPropagation()
        onClick()
      }}
      style={{
        ...controlButtonStyle,
        background: primary ? '#2563eb' : '#ffffff',
        color: primary ? '#ffffff' : '#334155',
      }}
    >
      {children}
    </button>
  )
}

function rangePosition(startFrame: number, endFrame: number, maxFrame: number) {
  const left = framePercent(startFrame, maxFrame)
  const right = framePercent(endFrame, maxFrame)
  return {
    left: `${left}%`,
    width: `${Math.max(right - left, 0.6)}%`,
  }
}

function framePercent(frame: number, maxFrame: number): number {
  return maxFrame === 0 ? 0 : (frame / maxFrame) * 100
}

function formatTime(frame: number, frameDurationMs: number): string {
  const totalSeconds = Math.max(0, frame * frameDurationMs) / 1000
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = (totalSeconds % 60).toFixed(1).padStart(4, '0')
  return `${minutes}:${seconds}`
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum)
}

const pageStyle: CSSProperties = {
  minHeight: '100vh',
  background: '#f8fafc',
  color: '#111827',
  fontFamily: 'system-ui, sans-serif',
}

const emptyPageStyle: CSSProperties = {
  ...pageStyle,
  display: 'grid',
  placeItems: 'center',
}

const headerStyle: CSSProperties = {
  minHeight: '4.5rem',
  padding: '0 1.5rem',
  borderBottom: '1px solid #d7dde8',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '1rem',
  background: '#ffffff',
}

const phaseLabelStyle: CSSProperties = {
  marginLeft: '1rem',
  color: '#2563eb',
  fontWeight: 800,
}

const trialTabStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.45rem',
  border: '1px solid',
  borderRadius: '8px',
  padding: '0.55rem 0.8rem',
  font: 'inherit',
  fontSize: '0.85rem',
  fontWeight: 700,
  cursor: 'pointer',
}

const countBadgeStyle: CSSProperties = {
  minWidth: '1.15rem',
  height: '1.15rem',
  borderRadius: '999px',
  display: 'inline-grid',
  placeItems: 'center',
  padding: '0 0.2rem',
  background: '#2563eb',
  color: '#ffffff',
  fontSize: '0.7rem',
}

const contentStyle: CSSProperties = {
  minHeight: 'calc(100vh - 4.5rem)',
  display: 'grid',
  gridTemplateColumns: 'minmax(0, 3fr) minmax(20rem, 2fr)',
}

const viewerColumnStyle: CSSProperties = {
  minWidth: 0,
  padding: '1.5rem 2rem 2rem',
  display: 'flex',
  flexDirection: 'column',
  gap: '1rem',
  alignItems: 'center',
}

const introStyle: CSSProperties = {
  width: '100%',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'flex-start',
  gap: '1rem',
}

const timeBadgeStyle: CSSProperties = {
  flexShrink: 0,
  border: '1px solid #cbd5e1',
  borderRadius: '8px',
  padding: '0.6rem 0.8rem',
  background: '#ffffff',
  color: '#111827',
  fontVariantNumeric: 'tabular-nums',
  fontWeight: 700,
}

const gamePanelStyle: CSSProperties = {
  position: 'relative',
  minHeight: '18rem',
  width: '100%',
  display: 'grid',
  placeItems: 'center',
  overflow: 'auto',
  border: '1px solid #dbe3ee',
  borderRadius: '12px',
  padding: '1.25rem',
  background: '#ffffff',
  boxShadow: '0 8px 24px rgba(15, 23, 42, 0.06)',
  cursor: 'pointer',
}

const pausedOverlayStyle: CSSProperties = {
  position: 'absolute',
  inset: 0,
  display: 'grid',
  placeItems: 'center',
  pointerEvents: 'none',
  background: 'rgba(248, 250, 252, 0.62)',
  color: '#111827',
  fontSize: '1.1rem',
  fontWeight: 800,
}

const controlsPanelStyle: CSSProperties = {
  width: '100%',
  padding: '1.25rem',
  border: '1px solid #dbe3ee',
  borderRadius: '12px',
  background: '#ffffff',
  boxSizing: 'border-box',
  boxShadow: '0 8px 24px rgba(15, 23, 42, 0.05)',
}

const timelineStyle: CSSProperties = {
  position: 'relative',
  width: '100%',
  height: '2.25rem',
  cursor: 'crosshair',
  touchAction: 'none',
  userSelect: 'none',
}

const timelineTrackStyle: CSSProperties = {
  position: 'absolute',
  left: 0,
  right: 0,
  top: '0.75rem',
  height: '0.75rem',
  border: '1px solid #cbd5e1',
  borderRadius: '5px',
  background:
    'repeating-linear-gradient(90deg, #e2e8f0 0, #e2e8f0 4.8%, #cbd5e1 5%)',
}

const selectionRangeStyle: CSSProperties = {
  position: 'absolute',
  top: '0.58rem',
  height: '1.1rem',
  border: '1px solid',
  borderRadius: '4px',
  boxSizing: 'border-box',
}

const handleStyle: CSSProperties = {
  position: 'absolute',
  top: '-0.25rem',
  bottom: '-0.25rem',
  width: '0.55rem',
  borderRadius: '3px',
  background: '#2563eb',
  boxShadow: '0 0 0 1px rgba(255, 255, 255, 0.85)',
  cursor: 'ew-resize',
  touchAction: 'none',
  pointerEvents: 'auto',
}

const playheadStyle: CSSProperties = {
  position: 'absolute',
  top: '0.25rem',
  width: '3px',
  height: '2rem',
  transform: 'translateX(-50%)',
  borderRadius: '2px',
  background: '#1e3a8a',
  boxShadow: '0 0 0 2px rgba(255, 255, 255, 0.8)',
  pointerEvents: 'none',
}

const controlsRowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr auto 1fr',
  alignItems: 'center',
  gap: '1rem',
  marginTop: '0.7rem',
}

const frameLabelStyle: CSSProperties = {
  color: '#64748b',
  fontSize: '0.85rem',
  fontVariantNumeric: 'tabular-nums',
}

const controlButtonStyle: CSSProperties = {
  width: '2.45rem',
  height: '2.45rem',
  display: 'grid',
  placeItems: 'center',
  border: '1px solid #cbd5e1',
  borderRadius: '7px',
  padding: 0,
  font: 'inherit',
  fontWeight: 900,
  cursor: 'pointer',
}

const speedLabelStyle: CSSProperties = {
  justifySelf: 'end',
  display: 'flex',
  alignItems: 'center',
  gap: '0.5rem',
  color: '#64748b',
  fontSize: '0.85rem',
  fontWeight: 700,
}

const speedSelectStyle: CSSProperties = {
  border: '1px solid #cbd5e1',
  borderRadius: '6px',
  padding: '0.4rem 0.55rem',
  background: '#ffffff',
  color: '#111827',
  font: 'inherit',
}

const dragHintStyle: CSSProperties = {
  margin: '0.8rem 0 0',
  color: '#2563eb',
  fontSize: '0.85rem',
  textAlign: 'center',
}

const sidePanelStyle: CSSProperties = {
  minHeight: 0,
  padding: '1.5rem',
  borderLeft: '1px solid #d7dde8',
  display: 'flex',
  flexDirection: 'column',
  gap: '1.25rem',
  background: '#ffffff',
}

const panelEyebrowStyle: CSSProperties = {
  margin: '0 0 0.35rem',
  color: '#2563eb',
  fontSize: '0.8rem',
  fontWeight: 800,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
}

const segmentListStyle: CSSProperties = {
  flex: 1,
  minHeight: '12rem',
  overflowY: 'auto',
  display: 'flex',
  flexDirection: 'column',
  gap: '0.75rem',
}

const emptySegmentsStyle: CSSProperties = {
  minHeight: '11rem',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '0.5rem',
  border: '1px dashed #cbd5e1',
  borderRadius: '10px',
  color: '#64748b',
  background: '#f8fafc',
  textAlign: 'center',
}

const segmentCardStyle: CSSProperties = {
  padding: '1rem',
  border: '1px solid #dbe3ee',
  borderLeft: '4px solid #2563eb',
  borderRadius: '10px',
  background: '#ffffff',
  boxShadow: '0 4px 12px rgba(15, 23, 42, 0.05)',
  cursor: 'pointer',
}

const segmentTimeStyle: CSSProperties = {
  margin: '0.4rem 0 0',
  color: '#2563eb',
  fontVariantNumeric: 'tabular-nums',
  fontWeight: 700,
}

const deleteButtonStyle: CSSProperties = {
  alignSelf: 'flex-start',
  border: 0,
  background: 'transparent',
  color: '#64748b',
  fontSize: '1.25rem',
  cursor: 'pointer',
}

const segmentMetaStyle: CSSProperties = {
  marginTop: '0.8rem',
  paddingTop: '0.8rem',
  borderTop: '1px solid #e2e8f0',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: '0.75rem',
  color: '#64748b',
  fontSize: '0.8rem',
}

const replayButtonStyle: CSSProperties = {
  border: '1px solid #cbd5e1',
  borderRadius: '6px',
  padding: '0.4rem 0.55rem',
  background: '#f8fafc',
  color: '#334155',
  font: 'inherit',
  fontSize: '0.78rem',
  fontWeight: 700,
  cursor: 'pointer',
}

const panelActionsStyle: CSSProperties = {
  paddingTop: '1rem',
  borderTop: '1px solid #e2e8f0',
}

const nextButtonStyle: CSSProperties = {
  width: '100%',
  border: 0,
  borderRadius: '8px',
  padding: '0.8rem 1rem',
  background: '#2563eb',
  color: '#ffffff',
  font: 'inherit',
  fontWeight: 800,
  cursor: 'pointer',
}
