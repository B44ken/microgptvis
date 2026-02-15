// Autograd Value node
export class Value {
    data: number; grad: number = 0; _children: Value[]; _local_grads: number[]

    constructor(data: number, _children: Value[] = [], _local_grads: number[] = []) {
        this.data = data
        this._children = _children
        this._local_grads = _local_grads
    }

    add(other: Value | number) {
        const o = other instanceof Value ? other : new Value(other)
        return new Value(this.data + o.data, [this, o], [1, 1])
    }
    sub(other: Value | number) {
        const o = other instanceof Value ? other : new Value(other)
        return new Value(this.data - o.data, [this, o], [1, -1])
    }
    mul(other: Value | number) {
        const o = other instanceof Value ? other : new Value(other)
        return new Value(this.data * o.data, [this, o], [o.data, this.data])
    }
    div(other: Value | number) {
        const o = other instanceof Value ? other : new Value(other)
        return this.mul(o.pow(-1))
    }
    exp() {
        const out = Math.exp(this.data)
        return new Value(out, [this], [out])
    }
    pow = (e: number) => new Value(this.data ** e, [this], [e * this.data ** (e - 1)])
    log = () => new Value(Math.log(this.data), [this], [1 / this.data])
    neg = () => this.mul(-1)
    relu = () => new Value(this.data < 0 ? 0 : this.data, [this], [this.data > 0 ? 1 : 0])

    backward() {
        const topo: Value[] = []
        const visited = new Set<Value>()
        const build = (v: Value) => {
            if (visited.has(v)) return
            visited.add(v)
            for (const c of v._children) build(c)
            topo.push(v)
        }
        build(this)
        this.grad = 1.0
        for (let i = topo.length - 1; i >= 0; i--) {
            const v = topo[i]
            for (let j = 0; j < v._children.length; j++)
                v._children[j].grad += v._local_grads[j] * v.grad
        }
    }
}

// Helpers
const sum = (arr: Value[]) => arr.reduce((a, b) => a.add(b))
const zip = <A, B>(a: A[], b: B[]) => a.map((ai, i) => [ai, b[i]] as [A, B])

const linear = (x: Value[], w: Value[][]) => w.map(wo => sum(wo.map((wi, i) => wi.mul(x[i]))))

function softmax(logits: Value[]) {
    const max_val = Math.max(...logits.map(v => v.data))
    const exps = logits.map(v => v.sub(max_val).exp())
    const total = sum(exps)
    return exps.map(e => e.div(total))
}

function rmsnorm(x: Value[]) {
    const ms = sum(x.map(xi => xi.mul(xi))).mul(1 / x.length)
    const s = ms.add(1e-5).pow(-0.5)
    return x.map(xi => xi.mul(s))
}

const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())

type StateDict = Record<string, Value[][]>

export class MicroGPT {
    n_embd = 16
    n_head = 4
    n_layer = 1
    block_size = 16
    head_dim = 4
    scale: number
    vocab_size: number
    state_dict: StateDict = {}
    params: Value[] = []
    m_buf: Float64Array
    v_buf: Float64Array
    step_count = 0

    docs: string[] = []
    uchars: string[] = []
    stoi: Record<string, number> = {}
    itos: Record<number, string> = {}
    bos = 0

    static fromDocs(docs: string[]) {
        const uchars = [...new Set(docs.join(''))].sort()
        const m = new MicroGPT(uchars.length + 1)
        m.docs = docs
        m.uchars = uchars
        uchars.forEach((c, i) => { m.stoi[c] = i; m.itos[i] = c })
        m.bos = uchars.length
        return m
    }

    tokenize = (doc: string) => [this.bos, ...doc.split('').map(c => this.stoi[c]), this.bos]

    trainOnDoc(doc?: string) {
        doc ??= this.docs[Math.floor(Math.random() * this.docs.length)]
        return this.trainStep(this.tokenize(doc))
    }

    constructor(vocab_size: number) {
        this.vocab_size = vocab_size
        this.head_dim = Math.floor(this.n_embd / this.n_head)
        this.scale = 1 / this.head_dim ** 0.5

        const matrix = (nout: number, nin: number, std = 0.08) =>
            Array.from({ length: nout }, () =>
                Array.from({ length: nin }, () => {
                    const p = new Value(gauss() * std)
                    this.params.push(p)
                    return p
                }))

        this.state_dict['wte'] = matrix(vocab_size, this.n_embd)
        this.state_dict['wpe'] = matrix(this.block_size, this.n_embd)
        this.state_dict['lm_head'] = matrix(vocab_size, this.n_embd)

        for (let i = 0; i < this.n_layer; i++) {
            this.state_dict[`layer${i}.attn_wq`] = matrix(this.n_embd, this.n_embd)
            this.state_dict[`layer${i}.attn_wk`] = matrix(this.n_embd, this.n_embd)
            this.state_dict[`layer${i}.attn_wv`] = matrix(this.n_embd, this.n_embd)
            this.state_dict[`layer${i}.attn_wo`] = matrix(this.n_embd, this.n_embd)
            this.state_dict[`layer${i}.mlp_fc1`] = matrix(4 * this.n_embd, this.n_embd)
            this.state_dict[`layer${i}.mlp_fc2`] = matrix(this.n_embd, 4 * this.n_embd)
        }

        this.m_buf = new Float64Array(this.params.length)
        this.v_buf = new Float64Array(this.params.length)
    }

    forward(token_id: number, pos_id: number, keys: Value[][][], values: Value[][][]) {
        const tok_emb = this.state_dict['wte'][token_id]
        const pos_emb = this.state_dict['wpe'][pos_id]
        let x = zip(tok_emb, pos_emb).map(([t, p]) => t.add(p))
        x = rmsnorm(x)

        for (let li = 0; li < this.n_layer; li++) {
            let x_residual = x
            x = rmsnorm(x)
            const q = linear(x, this.state_dict[`layer${li}.attn_wq`])
            const k = linear(x, this.state_dict[`layer${li}.attn_wk`])
            const v = linear(x, this.state_dict[`layer${li}.attn_wv`])
            keys[li].push(k)
            values[li].push(v)

            const x_attn: Value[] = []
            for (let h = 0; h < this.n_head; h++) {
                const hs = h * this.head_dim
                const q_h = q.slice(hs, hs + this.head_dim)
                const k_h = keys[li].map(ki => ki.slice(hs, hs + this.head_dim))
                const v_h = values[li].map(vi => vi.slice(hs, hs + this.head_dim))
                const attn_logits = k_h.map(kt => sum(zip(q_h, kt).map(([qi, ki]) => qi.mul(ki))).mul(this.scale))
                const attn_weights = softmax(attn_logits)
                for (let j = 0; j < this.head_dim; j++)
                    x_attn.push(sum(attn_weights.map((aw, t) => aw.mul(v_h[t][j]))))
            }

            x = linear(x_attn, this.state_dict[`layer${li}.attn_wo`])
            x = x.map((a, i) => a.add(x_residual[i]))

            x_residual = x
            x = rmsnorm(x)
            x = linear(x, this.state_dict[`layer${li}.mlp_fc1`])
            x = x.map(xi => xi.relu())
            x = linear(x, this.state_dict[`layer${li}.mlp_fc2`])
            x = x.map((a, i) => a.add(x_residual[i]))
        }

        const logits = linear(x, this.state_dict['lm_head'])
        return logits
    }

    newKV = () => Array.from({ length: this.n_layer }, () => [])

    trainStep(tokens: number[], lr = 0.01) {
        this.step_count++
        const beta1 = 0.85, beta2 = 0.99, eps = 1e-8
        for (const p of this.params) p.grad = 0

        const n = Math.min(this.block_size, tokens.length - 1)
        const keys = this.newKV()
        const values = this.newKV()
        const losses: Value[] = []

        for (let pos = 0; pos < n; pos++) {
            const logits = this.forward(tokens[pos], pos, keys, values)
            const probs = softmax(logits)
            losses.push(probs[tokens[pos + 1]].log().neg())
        }

        const loss = sum(losses).mul(1 / n)
        loss.backward()

        const lr_t = Math.max(lr / 1000, lr * (1 - this.step_count / 1000))
        const bc1 = 1 - beta1 ** this.step_count
        const bc2 = 1 - beta2 ** this.step_count
        for (let i = 0; i < this.params.length; i++) {
            const p = this.params[i]
            this.m_buf[i] = beta1 * this.m_buf[i] + (1 - beta1) * p.grad
            this.v_buf[i] = beta2 * this.v_buf[i] + (1 - beta2) * p.grad ** 2
            p.data -= lr_t * (this.m_buf[i] / bc1) / (Math.sqrt(this.v_buf[i] / bc2) + eps)
            p.grad = 0
        }
        return loss.data
    }

    generate(temperature = 0.5): string {
        const keys = this.newKV(), values = this.newKV()
        let token_id = this.bos
        const sample = []

        for (let pos = 0; pos < this.block_size; pos++) {
            const logits = this.forward(token_id, pos, keys, values)
            const probs = softmax(logits.map(l => l.div(temperature)))
            const r = Math.random()
            let cum = 0
            token_id = probs.length - 1
            for (let i = 0; i < probs.length; i++) {
                cum += probs[i].data
                if (r < cum) { token_id = i; break }
            }
            if (token_id === this.bos) break
            sample.push(this.uchars[token_id])
        }

        return sample.join('')
    }

    forwardTrace(token_id: number, pos_id: number, keys: Value[][][], values: Value[][][]): Trace {
        const d = (v: Value[]) => v.map(x => x.data)
        const tok_emb = this.state_dict['wte'][token_id]
        const pos_emb = this.state_dict['wpe'][pos_id]
        let x = zip(tok_emb, pos_emb).map(([t, p]) => t.add(p))
        const embedding = d(x)
        x = rmsnorm(x)
        const norm0 = d(x)

        const layerTraces: LayerTrace[] = []
        for (let li = 0; li < this.n_layer; li++) {
            let x_residual = x
            const r1 = d(x)
            x = rmsnorm(x)
            const norm1 = d(x)
            const q = linear(x, this.state_dict[`layer${li}.attn_wq`])
            const k = linear(x, this.state_dict[`layer${li}.attn_wk`])
            const v = linear(x, this.state_dict[`layer${li}.attn_wv`])
            keys[li].push(k)
            values[li].push(v)
            const cached_keys = keys[li].map(ki => ki.map(x => x.data))
            const cached_values = values[li].map(vi => vi.map(x => x.data))

            const heads: HeadTrace[] = []
            const x_attn: Value[] = []
            for (let h = 0; h < this.n_head; h++) {
                const hs = h * this.head_dim
                const q_h = q.slice(hs, hs + this.head_dim)
                const k_h = keys[li].map(ki => ki.slice(hs, hs + this.head_dim))
                const v_h = values[li].map(vi => vi.slice(hs, hs + this.head_dim))
                const attn_logits = k_h.map(kt => sum(zip(q_h, kt).map(([qi, ki]) => qi.mul(ki))).mul(this.scale))
                const attn_weights = softmax(attn_logits)
                const head_out: Value[] = []
                for (let j = 0; j < this.head_dim; j++)
                    head_out.push(sum(attn_weights.map((aw, t) => aw.mul(v_h[t][j]))))
                x_attn.push(...head_out)
                heads.push({ scores: d(attn_logits), weights: d(attn_weights), out: d(head_out) })
            }

            const concat = d(x_attn)
            x = linear(x_attn, this.state_dict[`layer${li}.attn_wo`])
            const wo = d(x)
            x = x.map((a, i) => a.add(x_residual[i]))
            const res1 = d(x)

            x_residual = x
            const r2 = d(x)
            x = rmsnorm(x)
            const norm2 = d(x)
            x = linear(x, this.state_dict[`layer${li}.mlp_fc1`])
            const fc1 = d(x)
            x = x.map(xi => xi.relu())
            const relu = d(x)
            x = linear(x, this.state_dict[`layer${li}.mlp_fc2`])
            const fc2 = d(x)
            x = x.map((a, i) => a.add(x_residual[i]))
            const res2 = d(x)

            layerTraces.push({ r1, norm1, q: d(q), k: d(k), v: d(v), cached_keys, cached_values, heads, concat, wo, res1, r2, norm2, fc1, relu, fc2, res2 })
        }

        const logits = linear(x, this.state_dict['lm_head'])
        const probs = softmax(logits)
        return { token_id, pos_id, embedding, norm0, layers: layerTraces, logits: d(logits), probs: d(probs) }
    }
}

export type HeadTrace = { scores: number[], weights: number[], out: number[] }
export type LayerTrace = {
    r1: number[], norm1: number[], q: number[], k: number[], v: number[],
    cached_keys: number[][], cached_values: number[][],
    heads: HeadTrace[], concat: number[], wo: number[], res1: number[],
    r2: number[], norm2: number[], fc1: number[], relu: number[], fc2: number[], res2: number[]
}
export type Trace = {
    token_id: number, pos_id: number, embedding: number[], norm0: number[],
    layers: LayerTrace[], logits: number[], probs: number[]
}

export { softmax }

