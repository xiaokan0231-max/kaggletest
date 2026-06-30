export const meta = {
  name: 'neurogolf-deepgolf-b6',
  description: 'Technique-primed deep ONNX-golf on rate-limited tasks: Equal/Where-as-output trick, small batch',
  phases: [{ title: 'DeepGolf', detail: 'apply proven floor-breaking technique; verify 262/262' }],
}

const REPO = '/Users/kanxiao/IdeaProjects/kaggletest'
const TARGETS = [
  { task: 'task382', dep: 13.93 }, { task: 'task234', dep: 13.96 },
  { task: 'task137', dep: 14.03 }, { task: 'task313', dep: 14.07 },
  { task: 'task097', dep: 14.11 }, { task: 'task203', dep: 14.12 },
  { task: 'task162', dep: 14.14 }, { task: 'task009', dep: 14.23 },
]

const SCHEMA = {
  type: 'object',
  required: ['task','rule','family','best_score','verified_ready','onnx_saved_path','notes'],
  properties: {
    task: { type: 'string' }, rule: { type: 'string' }, family: { type: 'string' },
    best_score: { type: 'number' }, verified_ready: { type: 'boolean' },
    onnx_saved_path: { type: 'string', description: 'rescue/ path if verified winner beats deployed, else empty' },
    notes: { type: 'string' },
  },
}

function prompt(t) {
  return `You are an EXPERT ONNX-golf engineer. Working dir: ${REPO}. Replace the fat-but-correct model for **${t.task}** (currently ${t.dep} pts) with a MUCH more compact functionally-equivalent ONNX, scoring as HIGH as possible. Serious iterative effort.

SCORING: points = max(1, 25 - ln(memory + params)). memory = sum over EVERY intermediate tensor of (num_elements x dtype_itemsize); **input and output tensors are EXCLUDED**. So the metric is dominated by interior [1,10,30,30] (=9000-elem) tensors. fp32=4B/elem, fp16=2, uint8/bool=1.

PROVEN FLOOR-BREAKING TECHNIQUE (two recent wins, task222->15.66, task193->15.49 — USE THIS):
1. Collapse the one-hot input to a single COLOR grid g [1,1,30,30] immediately: Conv with weight [1,10,1,1] = [[0,1,2,3,4,5,6,7,8,9]] (one f32 grid, 3600B — this is the irreducible input cost; do NOT slice/cast the 10-channel input first, that's strictly worse).
2. Do ALL interior logic on that single [1,1,30,30] grid using uint8/bool/float16 (tiny: 900-1800B each). Spatial ops: Conv (counts/sums), MaxPool (dilation/erosion), Slice+Pad (shifts), Equal/Greater/And/Or/Where (masks).
3. **Make the FINAL op that produces the 10-channel one-hot BE the graph 'output'** so it is EXCLUDED from memory — never materialize an interior [1,10,*] tensor. Two patterns:
   a. Compute the answer as a uint8 color grid gm [1,1,30,30]; Pad out-of-grid cells with sentinel 255; then a single \`Equal(gm_broadcast, colors[1,10,1,1] uint8 0..9)\` directly yields output one-hot bool [1,10,30,30] = the graph output.
   b. \`Where(cond[1,1,30,30] bool, input[1,10,30,30], e0[1,10,1,1] const)\` as the output (keeps input where cond true, constant one-hot elsewhere) — the only [1,10,30,30] tensor is the excluded output.

WORKFLOW:
1. Load+study: python with sys.path to ${REPO}/neurogolf/data/raw/neurogolf_utils, ng.load_examples(${parseInt(t.task.slice(4))}) -> {'train','test','arc-gen'}. Print grids, derive EXACT rule, confirm with numpy on ALL train+test.
2. If rule needs connected-component labeling / object counting / per-object identity -> set best_score=0, write rule+plan in notes, STOP (won't compress).
3. Else BUILD using the technique above. Save /tmp/b6_${t.task}.onnx. Crop to the common grid size if all examples share one (check first) to shrink tensors.
4. Verify: cd ${REPO} && python neurogolf_claude/tools/verify_local.py ${t.task} /tmp/b6_${t.task}.onnx -> "=== ${t.task}: READY=<bool> points=<float> ===".
5. ITERATE 3-5x to push score: uint8/bool everywhere, fewer ops, Equal/Where-as-output. Keep highest READY build.
6. If best READY > ${t.dep}: copy to ${REPO}/neurogolf_claude/rescue/${t.task}.onnx.

RULES: 0 unless passes official verify on ALL 262 cases. GENUINE algorithm only (never lookup — fails hidden test). Standard opset (domain ""), no Loop/Scan/If/Sequence. ALWAYS call StructuredOutput with your best honest result.`
}

phase('DeepGolf')
const results = await parallel(TARGETS.map(t => () => agent(prompt(t), { label: `dg6:${t.task}`, schema: SCHEMA })))
const ok = results.filter(Boolean)
const winners = ok.filter(r => r.verified_ready && r.onnx_saved_path)
log(`b6: ${winners.length} winners / ${ok.length} done: ${winners.map(w=>w.task+' '+w.best_score.toFixed(2)).join(', ')}`)
return { winners: winners.map(w=>({task:w.task,score:w.best_score,path:w.onnx_saved_path})),
  dossiers: ok.map(r=>({task:r.task,family:r.family,score:r.best_score,ready:r.verified_ready,rule:r.rule,notes:r.notes})) }
