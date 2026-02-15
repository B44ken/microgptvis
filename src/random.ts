export const seed = () => {}

export const choices = (choices: any[], probs: number[]) => {
    let r = Math.random();
    for (let i = 0; i < probs.length; i++)
        if ((r -= probs[i]) < 0) return choices[i];
    return choices[choices.length - 1];
}

export const gauss = (mu: number, sigma: number) => {
    let u = Math.random(), v = Math.random();
    return mu + sigma * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}