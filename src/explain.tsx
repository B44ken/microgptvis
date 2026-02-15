import type { Value } from './microgpt'

type StateDict = Record<string, Value[][]>

// returns flat array of .data from a weight matrix (first 16 rows flattened)
const flatWeights = (sd: StateDict, key: string): number[] | undefined => {
    const w = sd[key]
    if (!w) return undefined
    return w.slice(0, 16).flatMap(row => row.slice(0, 16).map(v => v.data))
}

const valColor = (v: number, min: number, max: number) => {
    if (v >= 0) {
        const t = max > 0 ? v / max : 0
        return `rgb(${187 * t}, ${204 * t}, ${255 * t})`
    }
    const t = min < 0 ? v / min : 0
    return `rgb(${238 * t}, ${68 * t}, ${68 * t})`
}

const WeightGrid = ({ data, label, rows = 16, cols = 16 }: { data: number[], label: string, rows?: number, cols?: number }) => {
    const min = Math.min(...data, -1), max = Math.max(...data, 1)
    return <div className='mt-2'>
        <div className='text-xs opacity-50 mb-1'>{label} [{rows}×{cols}]</div>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 6px)`, gap: 0 }}>
            {data.map((v, i) => <div key={i} title={v.toFixed(4)}
                style={{ width: 6, height: 6, background: valColor(v, min, max) }} />)}
        </div>
    </div>
}

type Entry = { text: string, io: string, weights?: string[] }

const explain: Record<string, Entry> = {
    'input': {
        text: 'The input is a single token index (0 to vocab_size). Each unique character in the training data gets its own index. Index 26 is the special BOS (beginning of sequence) token.',
        io: 'in: token string → out: int [1]',
    },
    'embed': {
        text: 'Look up two learned vectors: W_token[i] gives the character\'s semantic embedding, and W_pos[i] gives the positional embedding. Adding them produces a [16]-dim vector encoding both identity and position.',
        io: 'in: int [1] → out: float [16]',
        weights: ['wte', 'wpe'],
    },
    'norm0': {
        text: 'RMSNorm rescales the vector so its RMS value is 1. This stabilizes training by preventing activations from growing or shrinking uncontrollably. No learned weights — purely a normalization function.',
        io: 'in: float [16] → out: float [16]',
    },
    'transformer': {
        text: 'The transformer block is the core building block. It contains multi-head self-attention (to mix information across positions) followed by a feed-forward MLP (to process each position independently). Residual connections around both let gradients flow directly backward.',
        io: 'in: float [16] → out: float [16]',
    },
    'r1': {
        text: 'Save a copy of x before attention. After attention transforms x, we add this copy back (residual connection). This lets the model preserve information that might be lost during attention.',
        io: 'in: float [16] → out: float [16] (identity copy)',
    },
    'norm1': {
        text: 'RMSNorm before the attention projections. Normalizing keeps the Q, K, V projections operating on well-scaled inputs, which helps attention scores stay in a reasonable range for softmax.',
        io: 'in: float [16] → out: float [16]',
    },
    'qkv': {
        text: 'Three parallel linear projections produce query, key, and value vectors. Together they define what each position is "looking for" (Q), what it "contains" (K), and what information it "provides" (V).',
        io: 'in: float [16] → out: 3× float [16]',
        weights: ['layer0.attn_wq', 'layer0.attn_wk', 'layer0.attn_wv'],
    },
    'q': {
        text: 'Query projection: W_Q · x. Represents "what am I looking for?" Split into 4 heads of [4] dims each. Compared against cached keys via scaled dot-product to determine attention scores.',
        io: 'in: float [16] → out: float [16]',
        weights: ['layer0.attn_wq'],
    },
    'k': {
        text: 'Key projection: W_K · x. Represents "what do I contain?" Stored in the KV cache so future positions can attend back to this one. Each head uses a [4]-dim slice.',
        io: 'in: float [16] → out: float [16]',
        weights: ['layer0.attn_wk'],
    },
    'v': {
        text: 'Value projection: W_V · x. Represents "what information do I provide?" Once attention weights decide how much to attend to each position, values are weighted and summed to produce the output.',
        io: 'in: float [16] → out: float [16]',
        weights: ['layer0.attn_wv'],
    },
    'kv_cache': {
        text: 'The KV cache stores all previous keys and values. At position i, we have i key vectors and i value vectors. This is what makes autoregressive generation O(n) per step instead of O(n²) — we reuse previous computations.',
        io: 'in: float [16] (new k,v) → out: float [i × 16] (all cached)',
    },
    'cache_k': {
        text: 'The key cache accumulates key vectors from all previous positions. At position i, this is an [i × 16] matrix. Each head\'s [4]-dim slice is compared against the current query to compute attention scores.',
        io: 'in: float [16] → appended to: float [i × 16]',
    },
    'cache_v': {
        text: 'The value cache accumulates value vectors from all previous positions. After attention weights are computed from the keys, these cached values are weighted and summed to produce the attention output.',
        io: 'in: float [16] → appended to: float [i × 16]',
    },
    'heads': {
        text: 'Multi-head attention splits the [16]-dim space into 4 independent [4]-dim heads. Each head can learn different attention patterns (e.g., one might attend to recent tokens, another to similar characters). Results are concatenated back to [16].',
        io: 'in: q[16], k_cache[i×16], v_cache[i×16] → out: float [16]',
    },
    'head0': { text: 'Head 0 operates on dims [0:3]. It has its own independent attention pattern, learning which previous positions are relevant for this 4-dimensional subspace.', io: 'in: q[0:3], k[0:3], v[0:3] → out: float [4]' },
    'head1': { text: 'Head 1 operates on dims [4:7]. Independent from head 0, it can specialize in a different type of relationship between positions.', io: 'in: q[4:7], k[4:7], v[4:7] → out: float [4]' },
    'head2': { text: 'Head 2 operates on dims [8:11]. A third independent attention pattern over its own subspace of the embedding.', io: 'in: q[8:11], k[8:11], v[8:11] → out: float [4]' },
    'head3': { text: 'Head 3 operates on dims [12:15]. The fourth and final head, covering the last 4 dimensions of the embedding space.', io: 'in: q[12:15], k[12:15], v[12:15] → out: float [4]' },
    'scores0': { text: 'Raw dot products between this head\'s query slice and all cached key slices, scaled by 1/√d. Higher scores mean stronger similarity between the current query and a past key.', io: 'in: q[4], k_cache[i×4] → out: float [i]' },
    'scores1': { text: 'Raw dot products between this head\'s query slice and all cached key slices, scaled by 1/√d. Higher scores mean stronger similarity between the current query and a past key.', io: 'in: q[4], k_cache[i×4] → out: float [i]' },
    'scores2': { text: 'Raw dot products between this head\'s query slice and all cached key slices, scaled by 1/√d. Higher scores mean stronger similarity between the current query and a past key.', io: 'in: q[4], k_cache[i×4] → out: float [i]' },
    'scores3': { text: 'Raw dot products between this head\'s query slice and all cached key slices, scaled by 1/√d. Higher scores mean stronger similarity between the current query and a past key.', io: 'in: q[4], k_cache[i×4] → out: float [i]' },
    'weights0': { text: 'Softmax normalizes the raw scores into a probability distribution over all positions. Each weight is between 0 and 1, and they sum to 1. This determines how much each past position contributes to the output.', io: 'in: float [i] → out: float [i] (sums to 1)' },
    'weights1': { text: 'Softmax normalizes the raw scores into a probability distribution over all positions. Each weight is between 0 and 1, and they sum to 1. This determines how much each past position contributes to the output.', io: 'in: float [i] → out: float [i] (sums to 1)' },
    'weights2': { text: 'Softmax normalizes the raw scores into a probability distribution over all positions. Each weight is between 0 and 1, and they sum to 1. This determines how much each past position contributes to the output.', io: 'in: float [i] → out: float [i] (sums to 1)' },
    'weights3': { text: 'Softmax normalizes the raw scores into a probability distribution over all positions. Each weight is between 0 and 1, and they sum to 1. This determines how much each past position contributes to the output.', io: 'in: float [i] → out: float [i] (sums to 1)' },
    'attn_out0': { text: 'Weighted sum of cached value vectors using the attention weights. This is the "output" of this attention head — a [4]-dim vector that combines information from all attended positions.', io: 'in: weights[i], v_cache[i×4] → out: float [4]' },
    'attn_out1': { text: 'Weighted sum of cached value vectors using the attention weights. This is the "output" of this attention head — a [4]-dim vector that combines information from all attended positions.', io: 'in: weights[i], v_cache[i×4] → out: float [4]' },
    'attn_out2': { text: 'Weighted sum of cached value vectors using the attention weights. This is the "output" of this attention head — a [4]-dim vector that combines information from all attended positions.', io: 'in: weights[i], v_cache[i×4] → out: float [4]' },
    'attn_out3': { text: 'Weighted sum of cached value vectors using the attention weights. This is the "output" of this attention head — a [4]-dim vector that combines information from all attended positions.', io: 'in: weights[i], v_cache[i×4] → out: float [4]' },
    'wo': {
        text: 'The output projection W_O mixes the concatenated head outputs back together. This is the only place where information from different heads can interact. The result is a [16]-dim vector.',
        io: 'in: float [16] (concat heads) → out: float [16]',
        weights: ['layer0.attn_wo'],
    },
    'residual1': {
        text: 'The first residual block: add the attention output back to the saved R₁, then save the result as R₂ for the next residual. Skip connections let gradients flow directly backward through the network.',
        io: 'in: attn_out[16] + R₁[16] → out: float [16]',
    },
    'res1': {
        text: 'First residual connection: x = attention_output + R₁. This creates a skip connection so gradients flow directly backward. Without residuals, deep transformers are nearly impossible to train.',
        io: 'in: float [16] + float [16] → out: float [16]',
    },
    'r2': {
        text: 'Save x again before the MLP. Same idea as R₁ — after the MLP transforms x, we add this copy back so pre-MLP information is preserved.',
        io: 'in: float [16] → out: float [16] (identity copy)',
    },
    'norm2': {
        text: 'RMSNorm before the feed-forward MLP. Ensures the MLP receives well-scaled inputs regardless of how much attention amplified or reduced certain dimensions.',
        io: 'in: float [16] → out: float [16]',
    },
    'mlp': {
        text: 'A 2-layer feed-forward network applied independently to each position. It first projects up to [64] dims (4× expansion), applies ReLU nonlinearity, then projects back to [16]. This is where "thinking" happens — attention routes information, MLP processes it.',
        io: 'in: float [16] → out: float [16]',
        weights: ['layer0.mlp_fc1', 'layer0.mlp_fc2'],
    },
    'fc1': {
        text: 'The first MLP layer projects from [16] to [64] dimensions (4× expansion). This expansion gives the network more capacity to learn complex nonlinear transformations.',
        io: 'in: float [16] → out: float [64]',
        weights: ['layer0.mlp_fc1'],
    },
    'relu': {
        text: 'ReLU(x) = max(0, x). This activation function zeroes out negative values, introducing the nonlinearity that makes the network capable of learning complex functions. Without it, stacking linear layers would just produce another linear layer.',
        io: 'in: float [64] → out: float [64] (no learned weights)',
    },
    'fc2': {
        text: 'The second MLP layer projects back from [64] to [16] dimensions. Together with fc1, this forms a bottleneck: expand → nonlinearity → compress, which is thought to let the network learn diverse features in the expanded space.',
        io: 'in: float [64] → out: float [16]',
        weights: ['layer0.mlp_fc2'],
    },
    'res2': {
        text: 'Second residual connection: x = MLP_output + R₂. Now x contains contributions from the original input, attention, and the MLP. Each residual lets the model learn incremental refinements.',
        io: 'in: float [16] + float [16] → out: float [16]',
    },
    'logits': {
        text: 'The language model head W_lm projects from [16] to [vocab_size] unnormalized log-probabilities. Each logit corresponds to one character. Higher values mean the model predicts that character is more likely next.',
        io: 'in: float [16] → out: float [vocab_size]',
        weights: ['lm_head'],
    },
    'probs': {
        text: 'Softmax converts logits to a proper probability distribution: P(token_i) = exp(logit_i) / Σ exp(logit_j). The result sums to 1.0. During generation, we sample from this distribution to pick the next token.',
        io: 'in: float [vocab_size] → out: float [vocab_size] (sums to 1)',
    },
    'output': {
        text: 'Sample one token from the probability distribution. We draw a random number and walk through the cumulative distribution until we find the chosen token. The sampled token becomes the input for the next step of generation.',
        io: 'in: float [vocab_size] → out: int [1] (token index)',
    },
}

const faq: { q: string, a: string }[] = [
    {
        q: 'Why 16 dims, 4 heads, 64 in the MLP?',
        a: 'These are hyperparameters — arbitrary choices that trade off capacity vs speed. 16 dims is tiny (GPT-2 uses 768). 4 heads means 4 independent attention patterns. The MLP\'s 4× expansion (16→64→16) is standard in transformers and gives the network more room to learn features.',
    },
    {
        q: 'What does a cell in the weight heatmap mean?',
        a: 'Each row is one output neuron. Each column is one input. The cell at [row r, col c] is the learned strength of the connection from input c to output r. Blue = positive weight, red = negative, black = near zero. The matrix is a linear transformation: output = W · input.',
    },
    {
        q: 'How does training change the weights?',
        a: 'Each training step: (1) run a forward pass on a name, (2) compute cross-entropy loss (how wrong the predictions were), (3) backpropagation computes gradients — how much each weight contributed to the error, (4) Adam optimizer nudges each weight to reduce the loss. Over thousands of steps, the weights converge to patterns that predict characters well.',
    },
    {
        q: 'What does RMSNorm actually compute?',
        a: 'RMSNorm(x) = x / RMS(x), where RMS = √(mean(x²)). It rescales the vector so its root-mean-square equals 1. This prevents "activation drift" where values grow or shrink across layers. Unlike LayerNorm, it doesn\'t subtract the mean — simpler and works just as well.',
    },
    {
        q: 'Why scale attention scores by 1/√d?',
        a: 'Dot products between random vectors grow with dimension d. Without scaling, softmax would become extremely peaked (nearly one-hot) for large d, killing gradients. Dividing by √d keeps the variance of scores ≈ 1 regardless of head dimension.',
    },
    {
        q: 'Why not always pick the highest probability token?',
        a: 'Greedy decoding (argmax) produces repetitive, boring output. Sampling with temperature introduces controlled randomness: temperature < 1 sharpens the distribution (more deterministic), temperature > 1 flattens it (more creative). This model uses temperature = 0.5.',
    },
    {
        q: 'What is this model actually learning about names?',
        a: 'Character-level patterns: common starting letters, consonant-vowel alternation, typical endings (-son, -ley, -na). The attention heads learn which previous characters are relevant for predicting the next one. After enough training, it generates plausible-sounding names it has never seen.',
    },
    {
        q: 'What would more layers do?',
        a: 'More layers = more sequential processing steps. Layer 1 might learn character bigrams, layer 2 might learn trigrams and word structure. GPT-2 has 12-48 layers. This model uses 1 layer, so it can only learn relatively simple patterns — but it\'s enough for names.',
    },
    {
        q: 'Why do residual connections matter so much?',
        a: 'Without residuals, gradients must pass through every layer\'s transformations during backprop, often vanishing or exploding. Residuals add a "highway" where gradients flow directly backward. They also let each layer learn a small delta rather than reconstructing the entire representation.',
    },
]

export default ({ clicked, stateDict, onBack }: { clicked: string, stateDict?: StateDict, onBack?: () => void }) => {
    const entry = clicked ? explain[clicked] : undefined

    if (!entry) return <div className='mt-4 text-left'>
        <p className='text-sm opacity-50 mb-3'>click a node to learn more</p>
        <h3 className='font-bold text-sm mb-2'>FAQ</h3>
        {faq.map((f, i) => <details key={i} className='mb-2 text-sm'>
            <summary className='cursor-pointer opacity-75 hover:opacity-100'>{f.q}</summary>
            <p className='opacity-60 mt-1 pl-2 border-l border-zinc-700'>{f.a}</p>
        </details>)}
    </div>

    return <div className='text-left mt-4'>
        <button className='text-xs opacity-50 hover:opacity-100 mb-2' onClick={onBack}>← back</button>
        <h2 className='font-bold'>{clicked}</h2>
        <p className='text-sm opacity-75 mt-1'>{entry.text}</p>
        <p className='text-xs font-mono opacity-50 mt-2'>{entry.io}</p>
        {stateDict && entry.weights?.map(key => {
            const data = flatWeights(stateDict, key)
            if (!data) return null
            const w = stateDict[key]
            return <WeightGrid key={key} data={data} label={key}
                rows={Math.min(w.length, 16)} cols={Math.min(w[0].length, 16)} />
        })}
    </div>
}