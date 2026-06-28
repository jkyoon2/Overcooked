import { useEffect, useRef, useState } from 'react'
import type { GameState } from '../App'

const UNSCALED = 15
const SCALE = 4
const TILE = UNSCALED * SCALE

// Grids match zsceval/envs/overcooked_new/.../data/layouts/*.layout exactly.
// '1' and '2' are player start positions — rendered as floor.
const LAYOUT_TERRAIN: Record<string, string[]> = {
  '1_forced_easy': [
    'XXXXXXXXXXXXX',
    'X 1   P   2 X',
    'X     X     X',
    'O     X     D',
    'T     X     X',
    'X     S     X',
    'XXXXXXXXXXXXX',
  ],
  '1_forced_hard': [
    'XXXXXXXXXXXXX',
    'O 1   X   2 X',
    'X     X     X',
    'T     S     P',
    'X     X     X',
    'D     X     X',
    'XXXXXXXXXXXXX',
  ],
  '1_incentivized_easy': [
    'XXXXXXXXXXXXX',
    'X 1   P   2 X',
    'X     X     X',
    'O           D',
    'T     X     X',
    'X     S     X',
    'XXXXXXXXXXXXX',
  ],
  '1_incentivized_hard': [
    'XXXXXXXXXXXXX',
    'O 1       2 X',
    'X     X     X',
    'T     S     P',
    'X     X     X',
    'D           X',
    'XXXXXXXXXXXXX',
  ],
}

type Frame = readonly [number, number]

const TERRAIN: Record<string, Frame> = {
  counter:  [1, 1], dishes: [18, 1], floor:   [35, 1],
  onions:   [52, 1], pot:  [69, 1], serve:   [86, 1], tomatoes: [103, 1],
}
const TILE_TO_TERRAIN: Record<string, string> = {
  X: 'counter', ' ': 'floor', P: 'pot',
  O: 'onions',  T: 'tomatoes', D: 'dishes', S: 'serve',
  '1': 'floor', '2': 'floor',   // player start positions
}

const CHEFS: Record<string, Frame> = {
  'EAST': [1, 1],         'EAST-bluehat': [18, 1],
  'EAST-dish': [35, 1],   'EAST-greenhat': [52, 1],
  'EAST-onion': [69, 1],  'EAST-orangehat': [86, 1],
  'EAST-purplehat': [1, 18], 'EAST-redhat': [18, 18],
  'EAST-soup-onion': [35, 18], 'EAST-soup-tomato': [52, 18],
  'EAST-tomato': [69, 18],
  'NORTH': [86, 18],       'NORTH-bluehat': [1, 35],
  'NORTH-dish': [18, 35],  'NORTH-greenhat': [35, 35],
  'NORTH-onion': [52, 35], 'NORTH-orangehat': [69, 35],
  'NORTH-purplehat': [86, 35], 'NORTH-redhat': [1, 52],
  'NORTH-soup-onion': [18, 52], 'NORTH-soup-tomato': [35, 52],
  'NORTH-tomato': [52, 52],
  'SOUTH': [69, 52],       'SOUTH-bluehat': [86, 52],
  'SOUTH-dish': [1, 69],   'SOUTH-greenhat': [18, 69],
  'SOUTH-onion': [35, 69], 'SOUTH-orangehat': [52, 69],
  'SOUTH-purplehat': [69, 69], 'SOUTH-redhat': [86, 69],
  'SOUTH-soup-onion': [1, 86], 'SOUTH-soup-tomato': [18, 86],
  'SOUTH-tomato': [35, 86],
  'WEST': [52, 86],        'WEST-bluehat': [69, 86],
  'WEST-dish': [86, 86],   'WEST-greenhat': [103, 1],
  'WEST-onion': [103, 18], 'WEST-orangehat': [103, 35],
  'WEST-purplehat': [103, 52], 'WEST-redhat': [103, 69],
  'WEST-soup-onion': [103, 86], 'WEST-soup-tomato': [1, 103],
  'WEST-tomato': [18, 103],
}

const OBJECTS: Record<string, Frame> = {
  dish: [1, 1], onion: [18, 1], tomato: [239, 1],
}

const SOUPS: Record<string, Frame> = {
  'soup_cooked_tomato_0_onion_1': [0, 0],   'soup_cooked_tomato_0_onion_2': [15, 0],
  'soup_cooked_tomato_0_onion_3': [30, 0],  'soup_cooked_tomato_1_onion_0': [45, 0],
  'soup_cooked_tomato_1_onion_1': [60, 0],  'soup_cooked_tomato_1_onion_2': [75, 0],
  'soup_cooked_tomato_2_onion_0': [90, 0],  'soup_cooked_tomato_2_onion_1': [105, 1],
  'soup_cooked_tomato_3_onion_0': [120, 0],
  'soup_done_tomato_0_onion_1': [135, 0],   'soup_done_tomato_0_onion_2': [150, 0],
  'soup_done_tomato_0_onion_3': [165, 0],   'soup_done_tomato_1_onion_0': [180, 0],
  'soup_done_tomato_1_onion_1': [195, 0],   'soup_done_tomato_1_onion_2': [210, 1],
  'soup_done_tomato_2_onion_0': [225, 0],   'soup_done_tomato_2_onion_1': [240, 0],
  'soup_done_tomato_3_onion_0': [255, 0],
  'soup_idle_tomato_0_onion_1': [270, 0],   'soup_idle_tomato_0_onion_2': [285, 0],
  'soup_idle_tomato_0_onion_3': [300, 1],   'soup_idle_tomato_1_onion_0': [315, 0],
  'soup_idle_tomato_1_onion_1': [330, 0],   'soup_idle_tomato_1_onion_2': [345, 0],
  'soup_idle_tomato_2_onion_0': [360, 0],   'soup_idle_tomato_2_onion_1': [375, 0],
  'soup_idle_tomato_3_onion_0': [390, 1],
}

type ObjectSnapshot = GameState['objects'][number]
type PlayerSnapshot = GameState['players'][number]

function orientationToDir(o: readonly [number, number]): string {
  const [dx, dy] = o
  if (dy === -1) return 'NORTH'
  if (dy === 1)  return 'SOUTH'
  if (dx === 1)  return 'EAST'
  return 'WEST'
}

function normalizeHat(hat: string): string {
  const lc = hat.toLowerCase()
  for (const c of ['blue', 'green', 'red', 'orange', 'purple']) {
    if (lc.includes(c)) return `${c}hat`
  }
  return 'bluehat'
}

function heldSuffix(p: PlayerSnapshot): string {
  if (!p.held_object) return ''
  if (p.held_object === 'soup') {
    const t = (p.held_object_ingredients ?? []).filter(i => i === 'tomato').length
    const o = (p.held_object_ingredients ?? []).filter(i => i === 'onion').length
    return t >= o ? '-soup-tomato' : '-soup-onion'
  }
  return `-${p.held_object}`
}

function soupKey(s: ObjectSnapshot): string | null {
  const ings = s.ingredients ?? []
  const nT = ings.filter(i => i === 'tomato').length
  const nO = ings.filter(i => i === 'onion').length
  if (nT + nO === 0) return null
  const status = s.soup_state === 'ready' ? 'cooked' : 'idle'
  return `soup_${status}_tomato_${nT}_onion_${nO}`
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  imgs: Record<string, HTMLImageElement>,
  state: GameState,
  terrain: string[],
  hat0: string,
  hat1: string,
): void {
  ctx.imageSmoothingEnabled = false
  const blit = (img: HTMLImageElement, [sx, sy]: Frame, col: number, row: number) =>
    ctx.drawImage(img, sx, sy, UNSCALED, UNSCALED, col * TILE, row * TILE, TILE, TILE)

  const objAt = new Map<string, ObjectSnapshot>()
  for (const obj of state.objects) {
    objAt.set(`${obj.position[0]},${obj.position[1]}`, obj)
  }

  const hatColors = [normalizeHat(hat0), normalizeHat(hat1)]

  for (let row = 0; row < terrain.length; row++) {
    for (let col = 0; col < terrain[row].length; col++) {
      const tile = terrain[row][col]
      blit(imgs.terrain, TERRAIN[TILE_TO_TERRAIN[tile] ?? 'floor'], col, row)
      const obj = objAt.get(`${col},${row}`)
      if (!obj) continue
      if (obj.name === 'soup') {
        const key = soupKey(obj)
        const f = key ? SOUPS[key] : null
        if (f) blit(imgs.soups, f, col, row)
      } else {
        const f = OBJECTS[obj.name]
        if (f) blit(imgs.objects, f, col, row)
      }
    }
  }

  for (let i = 0; i < state.players.length; i++) {
    const p = state.players[i]
    const [col, row] = p.position
    const dir = orientationToDir(p.orientation as [number, number])
    const bodyKey = `${dir}${heldSuffix(p)}`
    const hatKey = `${dir}-${hatColors[i]}`
    const bodyF = CHEFS[bodyKey] ?? CHEFS[dir]
    const hatF = CHEFS[hatKey] ?? CHEFS[`${dir}-bluehat`]
    if (bodyF) blit(imgs.chefs, bodyF, col, row)
    if (hatF)  blit(imgs.chefs, hatF,  col, row)
  }
}

type Props = { state: GameState; hat0: string; hat1: string }

export default function GameView({ state, hat0, hat1 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgsRef = useRef<Record<string, HTMLImageElement>>({})
  const [imgsReady, setImgsReady] = useState(false)

  useEffect(() => {
    const srcs: Record<string, string> = {
      terrain: '/sprites/terrain.png', chefs: '/sprites/chefs.png',
      objects: '/sprites/objects.png', soups:  '/sprites/soups.png',
    }
    let done = 0
    const total = Object.keys(srcs).length
    for (const [key, src] of Object.entries(srcs)) {
      const img = new Image()
      img.onload = () => { imgsRef.current[key] = img; if (++done === total) setImgsReady(true) }
      img.src = src
    }
  }, [])

  useEffect(() => {
    if (!imgsReady) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const terrain = LAYOUT_TERRAIN[state.layout_name] ?? LAYOUT_TERRAIN['1_forced_easy']
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    drawGrid(ctx, imgsRef.current, state, terrain, hat0, hat1)
  }, [state, hat0, hat1, imgsReady])

  const terrain = LAYOUT_TERRAIN[state.layout_name] ?? LAYOUT_TERRAIN['1_forced_easy']
  const w = terrain[0].length * TILE
  const h = terrain.length * TILE

  return (
    <canvas
      ref={canvasRef}
      width={w}
      height={h}
      style={{ imageRendering: 'pixelated', display: 'block', width: '100%', height: 'auto', maxWidth: `${w}px` }}
    />
  )
}
