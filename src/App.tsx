import { useState, useEffect, useRef, useCallback } from 'react'
import * as tf from '@tensorflow/tfjs'
import { MicroGPT, type Trace, type ModelOpts } from './microgpt'
import Architecture from './arch'
import Explainer from './explain'
import names from './firstnames.json'

const defaults: Record<string, number> = { n_embd: 16, n_head: 4, n_layer: 1, block_size: 32, batch: 10 }

export default function App() {
  const [state, setState] = useState('idle')
  const [metrics, setMetrics] = useState({ loss: 0, step: 0 })
  const [output, setOutput] = useState('')
  const [trace, setTrace] = useState<Trace>()
  const [clicked, setClicked] = useState('')
  const [opts, setOpts] = useState(defaults)
  const [collapsed, setCollapsed] = useState(false)
  const [sidebar, setShowSidebar] = useState(false)
  const model = useRef<MicroGPT | null>(null)
  const gen = useRef<{ tokens: number[], token: number, pos: number, chars: string[] } | null>(null)

  const m = model.current

  const resetModel = async (newOpts: Record<string, number>) => {
    setState('idle')
    setOpts(newOpts)
    setMetrics({ loss: 0, step: 0 })
    setOutput('')
    setTrace(undefined)
    gen.current = null
    await tf.ready()
    model.current = await MicroGPT.fromDocs(names as string[], newOpts)
    console.log('model ready, backend:', tf.getBackend())
  }
  const setOpt = (key: keyof ModelOpts, val: number) => resetModel({ ...opts, [key]: val })
  const train = useCallback(() =>
    model.current && setMetrics({ loss: model.current.trainSteps(10, opts.batch), step: model.current.step_count })
    , [opts.batch])

  const genStep = useCallback(() => {
    if (!model.current) return

    if (!gen.current) gen.current = { tokens: [], token: model.current.bos, pos: 0, chars: [] }
    const g = gen.current
    if (g.pos >= model.current.block_size) { setState('idle'); return }

    const t = model.current.forwardTrace(g.token, g.pos, g.tokens)
    setTrace(t)

    const probs = t.probs
    let cum = 0, next = probs.length - 1
    const r = Math.random()
    for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (r < cum) { next = i; break } }
    if (next == model.current.bos) { setState('idle'); return }

    g.tokens.push(g.token)
    g.chars.push(model.current.uchars[next])
    g.token = next
    g.pos++
    setOutput(g.chars.join(''))
  }, [])

  useEffect(() => {
    let active = true
    const loop = async () => {
      if (!active) return
      if (state == 'gen') { genStep(); setTimeout(() => { if (active) loop() }, 500) }
      if (state == 'train') { await train(); if (active) requestAnimationFrame(loop) }
    }
    loop()
    return () => { active = false }
  }, [state, train, genStep])

  const toggleGen = () => {
    if (state == 'gen') setState('idle')
    else { gen.current = null; setOutput(''); setState('gen') }
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => void setTimeout(() => setOpt('n_embd', 16)), [])

  const Opt = ({ label, k }: { label: string, k: keyof ModelOpts }) =>
    <label className='flex justify-between'> {label} <input type='number' className='w-14 border rounded text-right' value={opts[k]} onChange={e => setOpt(k, +e.target.value)} /> </label>

  return <div className="flex flex-col md:flex-row h-screen text-sm mx-auto my-0 items-center justify-center">
    <a href='#' className="md:hidden fixed top-4 left-4 z-50 p-2" onClick={() => setShowSidebar(!sidebar)}>{sidebar ? 'x' : 'menu'}</a>

    <div className={`
      fixed inset-y-0 left-0 z-40 w-[90vw] bg-[#111] p-6 shadow-2xl transition-transform duration-300 transform
      ${sidebar ? 'translate-x-0' : '-translate-x-full'}
      md:translate-x-0 md:relative md:shadow-none md:p-0 md:py-12 md:flex md:flex-col md:h-full overflow-y-auto md:w-[360px] border-r md:border-none border-zinc-800
    `}>
      <h1 className="text-2xl font-bold mt-8 md:mt-0">microgpt</h1>
      <div className="flex gap-2 w-full *:flex-1 opacity-75 mt-4"> <p>step {metrics.step}</p> <p>loss {metrics.loss.toFixed(2)}</p> </div>

      <div className="flex gap-2 w-full *:py-1 *:flex-1 *:border *:rounded-sm *:px-1 mt-4">
        <button onClick={() => setState(state == 'train' ? 'idle' : 'train')}>{state == 'train' ? 'stop' : 'train'}</button>
        <button onClick={toggleGen}>{state == 'gen' ? 'stop' : 'generate'}</button>
      </div>

      <div className="mt-4">{output}_</div>

      <Explainer clicked={clicked} stateDict={m?.state_dict} onBack={() => setClicked('')} />

      <div className="grow"></div>

      <h3 className='font-bold mt-4 mb-2'>parameters</h3>
      <div className='flex flex-col gap-1'>
        <Opt label='dimension' k='n_embd' />
        <Opt label='layers' k='n_layer' />
        <Opt label='batch size' k='batch' />
        {/* <Opt label='context' k='block_size' /> */}
        {/* <Opt label='heads' k='n_head' /> */}
        <label className='flex justify-between'> collapse <input type='checkbox' checked={collapsed} onChange={e => setCollapsed(e.target.checked)} /> </label>
        <div className='opacity-75 text-right'> total params: {Math.floor((m?.num_params || 0) / 1000)}k </div>
      </div>
    </div>

    {/* Main Content (Architecture) */}
    <div className='h-screen md:w-auto w-full overflow-y-auto pt-16 md:pt-0'>
      <Architecture trace={trace} onClickNode={(id) => { setClicked(id); setShowSidebar(true) }} cfg={opts as Required<ModelOpts>} collapsed={collapsed} />
    </div>

  </div>
}