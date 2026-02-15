import { useState, useEffect } from 'react'
import { MicroGPT } from './microgpt'
import Architecture from './arch'
import names from './names'

const layers = ['input', 'embedding', 'rmsnorm', 'output']
const model = MicroGPT.fromDocs(names)

export default function App() {
  const [state, setState] = useState('idle')
  const [metrics, setMetrics] = useState({ loss: 0, step: 0 })
  const [output, setOutput] = useState('')

  const train = () => setMetrics({ loss: model.trainOnDoc(), step: model.step_count })

  useEffect(() => {
    const int = setInterval(() => (state == 'train') && train())
    return () => clearInterval(int)
  }, [state])

  return <>
    <h1>microgpt</h1>
    <span>step {metrics.step} | loss {metrics.loss.toFixed(4)}</span>
    <div>
      <button onClick={train}>step</button>
      <button onClick={() => setState(state == 'train' ? 'idle' : 'train')}>train</button>
    </div>

    <div> {output}_ </div>
    <button onClick={() => setOutput(model.generate())}>generate</button>

    {layers.map((label: string, i: number) => <div key={i}>{label}</div>)}

    <Architecture />
  </>
}
