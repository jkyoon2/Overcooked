type HatLegendProps = {
  playerHat: string
  aiHat: string
}

const HAT_COLORS: Record<string, string> = {
  blue: '#2563eb',
  red: '#dc2626',
  green: '#16a34a',
  yellow: '#ca8a04',
  purple: '#7c3aed',
  orange: '#ea580c',
  black: '#111827',
  white: '#f8fafc',
}

export default function HatLegend({ playerHat, aiHat }: HatLegendProps) {
  return (
    <div
      aria-label="Role hat legend"
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
        gap: '1rem',
        width: 'min(34rem, 100%)',
      }}
    >
      <HatItem label="You" hat={playerHat} />
      <HatItem label="AI" hat={aiHat} />
    </div>
  )
}

function HatItem({ label, hat }: { label: string; hat: string }) {
  const color = HAT_COLORS[hat.toLowerCase()] ?? hat
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        border: '1px solid #d7dde8',
        borderRadius: '8px',
        padding: '0.9rem 1rem',
        background: '#ffffff',
      }}
    >
      <span
        aria-hidden="true"
        style={{
          width: '1.6rem',
          height: '1.6rem',
          borderRadius: '999px 999px 6px 6px',
          background: color,
          border: hat.toLowerCase() === 'white' ? '1px solid #94a3b8' : '1px solid transparent',
          boxShadow: 'inset 0 -5px 0 rgba(0, 0, 0, 0.14)',
          flex: '0 0 auto',
        }}
      />
      <span style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', minWidth: 0 }}>
        <strong style={{ color: '#111827', fontSize: '1rem' }}>{label}</strong>
        <span style={{ color: '#475569', textTransform: 'capitalize' }}>{hat} hat</span>
      </span>
    </div>
  )
}
