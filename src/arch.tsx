import { type ReactNode, type ReactElement } from "react"

const colors = {
    attn: 'border-emerald-700 bg-emerald-950',
    mlp: 'border-orange-700 bg-orange-950',
    norm: 'border-purple-700 bg-purple-950',
}

const Size = ({ size = '' }: { size?: string }) => <div className="text-zinc-600 text-center">{size}</div>
const ArrowRow = ({ size = '\u00a0' }: { size?: string }) => <div className="text-zinc-600 text-center">{size}→</div>

const Node = ({ name, kind, size = '' }: { name: ReactNode, kind?: keyof typeof colors, size?: string }) =>
    <div className={`border rounded px-2 py-2 ${kind && colors[kind]}`}>{name}</div>

const Block = ({ label = '', children, row = false, dashed = false, size = '' }: { label?: string, children: ReactElement<any>[] | ReactElement<any> | string, dashed?: boolean, size?: string, row?: boolean }) =>
    <div className={`${dashed ? 'border border-dashed' : ''} border-zinc-700 rounded p-2 flex flex-col items-center gap-2`}>
        {label && <span className="opacity-75">{label}</span>}
        <div className={`${row ? 'flex-row' : 'flex-col'} flex items-center gap-2`}>
            {Array.isArray(children) ? children.map((c, i) => <div key={i}>{c}<Size size={c.props?.size} /></div>) : children}
        </div>
    </div >

const Head = ({ h }: { h: number }) => <Block label={<>x<sub>[{h * 4}:{h * 4 + 3}]</sub></>} dashed>
    <Node name={<>q<sub>[{h * 4}:{h * 4 + 3}]</sub> · k<sub>i[{h * 4}:{h * 4 + 3}]</sub></>} kind="attn" size="[i]" />
    <Node name="α = softmax" kind="attn" size="[i]" />
    <Node name={<>Σ α<sub>i</sub> · v<sub>i</sub>[{h * 4}:{h * 4 + 3}]</>} kind="attn" size="[4]" />
</Block>

const Transformer = () => <Block label="transformer block" dashed>
    <Node name={<>R<sub>1</sub> = x</>} size="[16]" />
    <Node name="RMSnorm" kind="norm" size="[16]" />
    <Block dashed row>
        <Node name={<>q = W<sub>Q</sub> · x</>} kind="attn" size="[16]" />
        <Node name={<>k = W<sub>K</sub> · x</>} kind="attn" size="[16]" />
        <Node name={<>v = W<sub>V</sub> · x</>} kind="attn" size="[16]" />
    </Block>
    <Block row dashed>
        <Node name={<>k<sub>i</sub> = cache<sub>K</sub> + k</>} size="[i × 16]" />
        <Node name={<>v<sub>i</sub> = cache<sub>V</sub> + v</>} size="[i × 16]" />
    </Block>
    <Block row>
        {[0, 1, 2, 3].map(h => <Head key={h} h={h} />)}
    </Block>
    {/* <Node name="concat" kind="attn" /> */}
    <Node name={<>W<sub>O</sub> · x<sub>[0:15]</sub></>} kind="attn" size="[16]" />
    <Block row dashed>
        <Node name={<>x = x + R<sub>1</sub></>} />
        <ArrowRow />
        <Node name={<>R<sub>2</sub> = x</>} />
    </Block>
    <Node name="RMSnorm" kind="norm" />
    <Block dashed label="Neural Network" row>
        <Node name={<>W<sub>1</sub> · x</>} kind="mlp" size="[64]" />
        <ArrowRow />
        <Node name="ReLU" kind="mlp" size="[64]" />
        <ArrowRow />
        <Node name={<>W<sub>2</sub> · x</>} kind="mlp" size="[16]" />
    </Block>
    <Node name={<>x = x + R<sub>2</sub></>} />
</Block>

export default () => <Block>
    <Node name="i = (input)" size="[1]" />
    <Node name={<>x = W<sub>token</sub>[i] + W<sub>pos</sub>[i]</>} size="[16]" />
    <Node name="RMSnorm" kind="norm" size="[16]" />

    <Transformer />

    <Node name={<>W<sub>lm</sub> · x</>} size="[v]" />
    <Node name="Softmax" size="[v]" />
    <div className="opacity-75">P(next char)</div>
</Block >