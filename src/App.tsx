import { useState, useEffect, useRef } from 'react'
import { MicroGPT, type Trace, type Value } from './microgpt'
import Architecture from './arch'
import Explainer from './explain'
import names from './names'

const model = MicroGPT.fromDocs(names)

export default function App() {
  const [state, setState] = useState('idle')
  const [metrics, setMetrics] = useState({ loss: 0, step: 0 })
  const [output, setOutput] = useState('')
  const [trace, setTrace] = useState<Trace>()
  const [clicked, setClicked] = useState('')
  const gen = useRef<{ keys: Value[][][], values: Value[][][], token: number, pos: number, chars: string[] } | null>(null)

  const train = () => {
    setMetrics({ loss: model.trainOnDoc(), step: model.step_count })
    const keys = model.newKV(), values = model.newKV()
    setTrace(model.forwardTrace(model.bos, 0, keys, values))
  }

  const genStep = () => {
    if (!gen.current) gen.current = { keys: model.newKV(), values: model.newKV(), token: model.bos, pos: 0, chars: [] }
    const g = gen.current
    if (g.pos >= model.block_size) { setState('idle'); return }
    const t = model.forwardTrace(g.token, g.pos, g.keys, g.values)
    setTrace(t)
    const probs = t.probs
    let cum = 0, next = probs.length - 1
    const r = Math.random()
    for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (r < cum) { next = i; break } }
    if (next === model.bos) { setState('idle'); return }
    g.chars.push(model.uchars[next])
    g.token = next
    g.pos++
    setOutput(g.chars.join(''))
  }

  useEffect(() => {
    const int = setInterval(() => {
      if (state == 'train') train()
      if (state == 'gen') genStep()
    }, state == 'gen' ? 1000 : 0)
    return () => clearInterval(int)
  }, [state])

  const toggleGen = () => {
    if (state == 'gen') setState('idle')
    else { gen.current = null; setOutput(''); setState('gen') }
  }

  return <div className="flex flex-row w-full h-screen pt-8">

    <div className='w-70'>
      <h1 className="text-2xl font-bold">microgpt</h1>
      {/* <span className="font-mono text-sm opacity-50">step {metrics.step} <br /> loss {metrics.loss.toFixed(3)}</span> */}
      <div className="flex gap-2 w-full *:flex-1 ">
        <span className="font-mono opacity-50">step {metrics.step}</span>
        <span className="font-mono opacity-50">loss {metrics.loss.toFixed(2)}</span>
      </div>

      <div className="flex gap-2 w-full *:flex-1 *:border *:rounded-sm *:px-1">
        <button onClick={() => setState(state == 'train' ? 'idle' : 'train')}>{state == 'train' ? 'stop' : 'train'}</button>
        <button onClick={toggleGen}>{state == 'gen' ? 'stop' : 'go'}</button>
      </div>

      <div className=''>{output}_</div>

      <Explainer clicked={clicked} stateDict={model.state_dict} onBack={() => setClicked('')} />
    </div>

    <div className='h-screen overflow-y-auto w-[700px]'>
      <Architecture trace={trace} onClickNode={setClicked} />
    </div>

  </div>
}
