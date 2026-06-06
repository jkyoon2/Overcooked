import { useEffect, useRef, useState } from 'react'
import type { GameState, ObjectSnapshot, PlayerSnapshot } from '../lib/websocket'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const UNSCALED = 15  // original sprite tile size (pixels in the PNG)
const SCALE = 4      // render at 4× → 60px per tile
const TILE = UNSCALED * SCALE  // 60

// ---------------------------------------------------------------------------
// Layout terrain registry  (player-start markers 1/2 stripped → floor ' ')
// ---------------------------------------------------------------------------

const GRID_13x5: string[] = [
  'XXXXXXXXXXXXX',
  'O   DTXTD   O',
  'XX    P    XX',
  'S     P     S',
  'XXXXXTXTXXXXX',
]

const LAYOUT_TERRAIN: Record<string, string[]> = {
  ttt: GRID_13x5,
  ooo: GRID_13x5,
  tto: GRID_13x5,
  too: GRID_13x5,
  corner_onion_tomato: [
    'XTTXXXX',
    'O     X',
    'O  XX X',
    'X X   X',
    'X X   D',
    'XXXSPPX',
  ],
  cramped_room: [
    'XXPXX',
    'O   O',
    'X   X',
    'XDXSX',
  ],
}

// ---------------------------------------------------------------------------
// Sprite frame coordinates  [sx, sy]  (all tiles are 15×15 in the source PNG)
// ---------------------------------------------------------------------------

type Frame = readonly [number, number]

// terrain.png  (1px padding between frames)
const TERRAIN: Record<string, Frame> = {
  counter:  [1, 1],
  dishes:   [18, 1],
  floor:    [35, 1],
  onions:   [52, 1],
  pot:      [69, 1],
  serve:    [86, 1],
  tomatoes: [103, 1],
}

const TILE_TO_TERRAIN: Record<string, string> = {
  X: 'counter', ' ': 'floor', P: 'pot',
  O: 'onions',  T: 'tomatoes', D: 'dishes', S: 'serve',
}

// chefs.png  (1px padding between frames)
const CHEFS: Record<string, Frame> = {
  'EAST':              [1, 1],   'EAST-bluehat':    [18, 1],
  'EAST-dish':         [35, 1],  'EAST-greenhat':   [52, 1],
  'EAST-onion':        [69, 1],  'EAST-orangehat':  [86, 1],
  'EAST-purplehat':    [1, 18],  'EAST-redhat':     [18, 18],
  'EAST-soup-onion':   [35, 18], 'EAST-soup-tomato':[52, 18],
  'EAST-tomato':       [69, 18],
  'NORTH':             [86, 18], 'NORTH-bluehat':   [1, 35],
  'NORTH-dish':        [18, 35], 'NORTH-greenhat':  [35, 35],
  'NORTH-onion':       [52, 35], 'NORTH-orangehat': [69, 35],
  'NORTH-purplehat':   [86, 35], 'NORTH-redhat':    [1, 52],
  'NORTH-soup-onion':  [18, 52], 'NORTH-soup-tomato':[35, 52],
  'NORTH-tomato':      [52, 52],
  'SOUTH':             [69, 52], 'SOUTH-bluehat':   [86, 52],
  'SOUTH-dish':        [1, 69],  'SOUTH-greenhat':  [18, 69],
  'SOUTH-onion':       [35, 69], 'SOUTH-orangehat': [52, 69],
  'SOUTH-purplehat':   [69, 69], 'SOUTH-redhat':    [86, 69],
  'SOUTH-soup-onion':  [1, 86],  'SOUTH-soup-tomato':[18, 86],
  'SOUTH-tomato':      [35, 86],
  'WEST':              [52, 86], 'WEST-bluehat':    [69, 86],
  'WEST-dish':         [86, 86], 'WEST-greenhat':   [103, 1],
  'WEST-onion':        [103, 18],'WEST-orangehat':  [103, 35],
  'WEST-purplehat':    [103, 52],'WEST-redhat':     [103, 69],
  'WEST-soup-onion':   [103, 86],'WEST-soup-tomato':[1, 103],
  'WEST-tomato':       [18, 103],
}

// objects.png  (1px padding between frames)
const OBJECTS: Record<string, Frame> = {
  dish:   [1, 1],
  onion:  [18, 1],
  tomato: [239, 1],
}

// soups.png  (no padding — direct coordinates from soups.json)
// Key: soup_{status}_tomato_{n}_onion_{m}
const SOUPS: Record<string, Frame> = {
  'soup_cooked_tomato_0_onion_1': [0, 0],   'soup_cooked_tomato_0_onion_2': [15, 0],
  'soup_cooked_tomato_0_onion_3': [30, 0],  'soup_cooked_tomato_1_onion_0': [45, 0],
  'soup_cooked_tomato_1_onion_1': [60, 0],  'soup_cooked_tomato_1_onion_2': [75, 0],
  'soup_cooked_tomato_2_onion_0': [90, 0],  'soup_cooked_tomato_2_onion_1': [105, 1],
  'soup_cooked_tomato_3_onion_0': [120, 0],
  'soup_done_tomato_0_onion_1':   [135, 0], 'soup_done_tomato_0_onion_2':   [150, 0],
  'soup_done_tomato_0_onion_3':   [165, 0], 'soup_done_tomato_1_onion_0':   [180, 0],
  'soup_done_tomato_1_onion_1':   [195, 0], 'soup_done_tomato_1_onion_2':   [210, 1],
  'soup_done_tomato_2_onion_0':   [225, 0], 'soup_done_tomato_2_onion_1':   [240, 0],
  'soup_done_tomato_3_onion_0':   [255, 0],
  'soup_idle_tomato_0_onion_1':   [270, 0], 'soup_idle_tomato_0_onion_2':   [285, 0],
  'soup_idle_tomato_0_onion_3':   [300, 1], 'soup_idle_tomato_1_onion_0':   [315, 0],
  'soup_idle_tomato_1_onion_1':   [330, 0], 'soup_idle_tomato_1_onion_2':   [345, 0],
  'soup_idle_tomato_2_onion_0':   [360, 0], 'soup_idle_tomato_2_onion_1':   [375, 0],
  'soup_idle_tomato_3_onion_0':   [390, 1],
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function orientationToDir(orientation: readonly [number, number]): string {
  const [dx, dy] = orientation
  if (dy === -1) return 'NORTH'
  if (dy === 1)  return 'SOUTH'
  if (dx === 1)  return 'EAST'
  if (dx === -1) return 'WEST'
  return 'SOUTH'
}

function normalizeHat(hat: string): string {
  const lc = hat.toLowerCase()
  for (const c of ['blue', 'green', 'red', 'orange', 'purple']) {
    if (lc.includes(c)) return `${c}hat`
  }
  return 'bluehat'
}

function heldSuffix(player: PlayerSnapshot): string {
  if (!player.held_object) return ''
  if (player.held_object === 'soup') {
    const ingredients = player.held_object_ingredients ?? []
    const tomatoCount = ingredients.filter((item) => item === 'tomato').length
    const onionCount = ingredients.filter((item) => item === 'onion').length
    return tomatoCount >= onionCount ? '-soup-tomato' : '-soup-onion'
  }
  return `-${player.held_object}`  // '-onion', '-tomato', '-dish'
}

function soupFrameKey(soup: ObjectSnapshot): string | null {
  const ings = soup.ingredients ?? []
  const nT = ings.filter(i => i === 'tomato').length
  const nO = ings.filter(i => i === 'onion').length
  if (nT + nO === 0) return null
  const status = soup.soup_state === 'ready' ? 'cooked' : 'idle'
  return `soup_${status}_tomato_${nT}_onion_${nO}`
}

// ---------------------------------------------------------------------------
// Canvas draw pass
// ---------------------------------------------------------------------------

function drawGrid(
  ctx: CanvasRenderingContext2D,
  imgs: Record<string, HTMLImageElement>,
  state: GameState,
  terrain: string[],
  hat0: string,
  hat1: string,
): void {
  ctx.imageSmoothingEnabled = false

  const blit = (img: HTMLImageElement, [sx, sy]: Frame, col: number, row: number) => {
    ctx.drawImage(img, sx, sy, UNSCALED, UNSCALED, col * TILE, row * TILE, TILE, TILE)
  }

  // Index dynamic objects by "col,row"
  const objAt = new Map<string, ObjectSnapshot>()
  for (const obj of state.objects) {
    objAt.set(`${obj.position[0]},${obj.position[1]}`, obj)
  }

  // hat0=playerHat (index 0=human), hat1=aiHat (index 1=AI)
  const hatColors = [normalizeHat(hat0), normalizeHat(hat1)]

  // Pass 1: terrain + object overlays
  for (let row = 0; row < terrain.length; row++) {
    for (let col = 0; col < terrain[row].length; col++) {
      const tile = terrain[row][col]
      const tFrame = TERRAIN[TILE_TO_TERRAIN[tile] ?? 'floor']
      blit(imgs.terrain, tFrame, col, row)

      const obj = objAt.get(`${col},${row}`)
      if (!obj) continue

      if (obj.name === 'soup') {
        const key = soupFrameKey(obj)
        const f = key ? SOUPS[key] : null
        if (f) blit(imgs.soups, f, col, row)
      } else {
        const f = OBJECTS[obj.name]
        if (f) blit(imgs.objects, f, col, row)
      }
    }
  }

  // Pass 2: chefs (body + hat overlay — same visual shape per AC6)
  for (let i = 0; i < state.players.length; i++) {
    const p = state.players[i]
    const [col, row] = p.position
    const dir = orientationToDir(p.orientation as [number, number])
    const bodyKey = `${dir}${heldSuffix(p)}`
    const hatKey  = `${dir}-${hatColors[i]}`

    const bodyF = CHEFS[bodyKey] ?? CHEFS[dir]
    const hatF  = CHEFS[hatKey]  ?? CHEFS[`${dir}-bluehat`]
    if (bodyF) blit(imgs.chefs, bodyF, col, row)
    if (hatF)  blit(imgs.chefs, hatF, col, row)
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type GameViewProps = {
  state: GameState
  playerHat: string
  aiHat: string
}

export default function GameView({ state, playerHat, aiHat }: GameViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgsRef   = useRef<Record<string, HTMLImageElement>>({})
  const [imgsReady, setImgsReady] = useState(false)

  // Load sprite sheets once on mount
  useEffect(() => {
    const srcs: Record<string, string> = {
      terrain: '/sprites/terrain.png',
      chefs:   '/sprites/chefs.png',
      objects: '/sprites/objects.png',
      soups:   '/sprites/soups.png',
    }
    let done = 0
    const total = Object.keys(srcs).length
    for (const [key, src] of Object.entries(srcs)) {
      const img = new Image()
      img.onload = () => {
        imgsRef.current[key] = img
        if (++done === total) setImgsReady(true)
      }
      img.src = src
    }
  }, [])

  // Redraw on every state change (once images are ready)
  useEffect(() => {
    if (!imgsReady) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const terrain = LAYOUT_TERRAIN[state.layout_name] ?? GRID_13x5
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    // index 0=human, index 1=AI
    drawGrid(ctx, imgsRef.current, state, terrain, playerHat, aiHat)
  }, [state, playerHat, aiHat, imgsReady])

  const terrain = LAYOUT_TERRAIN[state.layout_name] ?? GRID_13x5
  const w = terrain[0].length * TILE
  const h = terrain.length    * TILE

  return (
    <canvas
      ref={canvasRef}
      width={w}
      height={h}
      style={{ imageRendering: 'pixelated', display: 'block' }}
    />
  )
}
