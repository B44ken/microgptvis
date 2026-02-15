import { type ReactNode, type ReactElement } from "react"
import type { Trace, LayerTrace, HeadTrace } from './microgpt'

// const colors = {
//     attn: 'border-emerald-700 bg-emerald-950',
//     mlp: 'border-orange-700 bg-orange-950',
//     norm: 'border-purple-700 bg-purple-950',
// }
const colors = { attn: '', mlp: '', norm: '' }

const valColor = (v: number, min: number, max: number) => {
    if (v >= 0) {
        const t = max > 0 ? v / max : 0
        return `rgb(${187 * t}, ${204 * t}, ${255 * t})`
    }
    const t = min < 0 ? v / min : 0
    return `rgb(${238 * t}, ${68 * t}, ${68 * t})`
}

const Val = ({ data, n = 16 }: { data?: number[], n?: number }) => {
    if (!data) return <div className="flex gap-0 w-fit mx-auto items-center mt-1 *:w-2 *:h-6 border border-[0.5]">
        {Array.from({ length: n }, (_, i) => <div key={i} style={{ background: '#111' }} />)}
    </div>
    const min = Math.min(...data, -1), max = Math.max(...data, 1)
    return <div className="flex gap-0 w-fit mx-auto items-center mt-1 *:w-2 *:h-6 border border-[0.5]">
        {data.slice(0, 16).map((v, i) => <div key={i} title={v.toFixed(3)} style={{ background: valColor(v, min, max) }} />)}
    </div>
}

const CacheGrid = ({ grid }: { grid?: number[][] }) => {
    if (!grid || grid.length === 0) return <div className="flex gap-0 w-fit mx-auto items-center mt-1 border border-[0.5]">
        {Array.from({ length: 16 }, (_, i) => <div key={i} style={{ width: 6, height: 6, background: '#111' }} />)}
    </div>
    const flat = grid.flat()
    const min = Math.min(...flat, -1), max = Math.max(...flat, 1)
    const cols = Math.min(grid[0].length, 16)
    return <div className="w-fit mx-auto mt-1 border border-[0.5]" style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 6px)` }}>
        {grid.map((row, ri) => row.slice(0, 16).map((v, ci) =>
            <div key={`${ri}-${ci}`} title={`pos ${ri}, dim ${ci}: ${v.toFixed(3)}`}
                style={{ width: 6, height: 6, background: valColor(v, min, max) }} />
        ))}
    </div>
}

const Size = ({ size = '' }: { size?: string }) => <div className="text-zinc-600 text-center">{size}</div>
const ArrowRow = ({ size = '\u00a0' }: { size?: string }) => <div className="text-zinc-600 text-center">{size}→</div>

const Node = ({ name, kind, size = '', data, grid, tooltip, onClick, n }: { name: ReactNode, kind?: keyof typeof colors, size?: string, data?: number[], grid?: number[][], tooltip?: string, onClick?: (id: string) => void, n?: number }) =>
    <div className={`border rounded px-2 py-2 ${kind && colors[kind]} ${tooltip ? 'cursor-pointer hover:border-zinc-400' : ''}`}
        onClick={e => { if (tooltip) { e.stopPropagation(); onClick?.(tooltip) } }}>
        {name}
        {grid ? <CacheGrid grid={grid} /> : <Val data={data} n={n} />}
    </div>

const Block = ({ label = '', children, row = false, dashed = false, size = '', tooltip, onClick }: { label?: string | ReactNode, children: ReactElement<any>[] | ReactElement<any> | string, dashed?: boolean, size?: string, row?: boolean, tooltip?: string, onClick?: (id: string) => void }) =>
    <div className={`${dashed ? 'border border-dashed' : ''} border-zinc-700 rounded p-2 flex flex-col items-center gap-2 text-sm ${tooltip ? 'cursor-pointer hover:border-zinc-400' : ''}`}
        onClick={e => { if (tooltip) { e.stopPropagation(); onClick?.(tooltip) } }}>
        {label && <span className="opacity-75">{label}</span>}
        <div className={`${row ? 'flex-row' : 'flex-col'} flex items-center gap-2`}>
            {Array.isArray(children) ? children.map((c, i) => <div key={i}>{c}<Size size={c.props?.size} /></div>) : children}
        </div>
    </div >

const Head = ({ h, t, onClick }: { h: number, t?: HeadTrace, onClick?: (id: string) => void }) => <Block label={<>head {h}</>} dashed tooltip={`head${h}`} onClick={onClick}>
    <Node name={<>q<sub>[{h * 4}:{h * 4 + 3}]</sub> · k<sub>i[{h * 4}:{h * 4 + 3}]</sub></>} kind="attn" size="[i]" data={t?.scores} tooltip={`scores${h}`} onClick={onClick} n={1} />
    <Node name="α = softmax" kind="attn" size="[i]" data={t?.weights} tooltip={`weights${h}`} onClick={onClick} n={1} />
    <Node name={<>Σ α<sub>i</sub> · v<sub>i</sub>[{h * 4}:{h * 4 + 3}]</>} kind="attn" size="[4]" data={t?.out} tooltip={`attn_out${h}`} onClick={onClick} n={4} />
</Block>

const Transformer = ({ t, onClick }: { t?: LayerTrace, onClick?: (id: string) => void }) => <Block label="transformer block" dashed tooltip='transformer' onClick={onClick}>
    <Node name={<>R<sub>1</sub> = x</>} size="[16]" data={t?.r1} tooltip='r1' onClick={onClick} />
    <Node name="RMSnorm" kind="norm" size="[16]" data={t?.norm1} tooltip='norm1' onClick={onClick} />
    <Block dashed row tooltip='qkv' onClick={onClick}>
        <Node name={<>q = W<sub>Q</sub> · x</>} kind="attn" size="[16]" data={t?.q} tooltip='q' onClick={onClick} />
        <Node name={<>k = W<sub>K</sub> · x</>} kind="attn" size="[16]" data={t?.k} tooltip='k' onClick={onClick} />
        <Node name={<>v = W<sub>V</sub> · x</>} kind="attn" size="[16]" data={t?.v} tooltip='v' onClick={onClick} />
    </Block>
    <Block row dashed tooltip='kv_cache' onClick={onClick}>
        <Node name={<>k<sub>i</sub> = cache<sub>K</sub> + k</>} size="[i × 16]" grid={t?.cached_keys} tooltip='cache_k' onClick={onClick} />
        <Node name={<>v<sub>i</sub> = cache<sub>V</sub> + v</>} size="[i × 16]" grid={t?.cached_values} tooltip='cache_v' onClick={onClick} />
    </Block>
    <Block row tooltip='heads' onClick={onClick}>
        {[0, 1, 2, 3].map(h => <Head key={h} h={h} t={t?.heads[h]} onClick={onClick} />)}
    </Block>
    <Node name={<>W<sub>O</sub> · x<sub>[0:15]</sub></>} kind="attn" size="[16]" data={t?.wo} tooltip='wo' onClick={onClick} />
    <Block row dashed tooltip='residual1' onClick={onClick}>
        <Node name={<>x = x + R<sub>1</sub></>} data={t?.res1} tooltip='res1' onClick={onClick} />
        <ArrowRow />
        <Node name={<>R<sub>2</sub> = x</>} data={t?.r2} tooltip='r2' onClick={onClick} />
    </Block>
    <Node name="RMSnorm" kind="norm" data={t?.norm2} tooltip='norm2' onClick={onClick} />
    <Block dashed label="neural net" row tooltip='mlp' onClick={onClick}>
        <Node name={<>W<sub>1</sub> · x</>} kind="mlp" size="[64]" data={t?.fc1} tooltip='fc1' onClick={onClick} />
        <ArrowRow />
        <Node name="ReLU" kind="mlp" size="[64]" data={t?.relu} tooltip='relu' onClick={onClick} />
        <ArrowRow />
        <Node name={<>W<sub>2</sub> · x</>} kind="mlp" size="[16]" data={t?.fc2} tooltip='fc2' onClick={onClick} />
    </Block>
    <Node name={<>x = x + R<sub>2</sub></>} data={t?.res2} tooltip='res2' onClick={onClick} />
</Block>

export default ({ trace, onClickNode }: { trace?: Trace, onClickNode?: (id: string) => void }) => <Block>
    <Node name="i = (input)" size="[1]" data={trace ? [trace.token_id] : undefined} tooltip='input' onClick={onClickNode} />
    <Node name={<>x = W<sub>token</sub>[i] + W<sub>pos</sub>[p]</>} size="[16]" data={trace?.embedding} tooltip='embed' onClick={onClickNode} />
    <Node name="RMSnorm" kind="norm" size="[16]" data={trace?.norm0} tooltip='norm0' onClick={onClickNode} />

    <Transformer t={trace?.layers[0]} onClick={onClickNode} />

    <Node name={<>W<sub>lm</sub> · x</>} size="[v]" data={trace?.logits} tooltip='logits' onClick={onClickNode} />
    <Node name="softmax" size="[v]" data={trace?.probs} tooltip='probs' onClick={onClickNode} />
    <Node name="output = sample(P)" size="[1]" data={trace ? [trace.token_id] : undefined} tooltip='output' onClick={onClickNode} />
</Block >