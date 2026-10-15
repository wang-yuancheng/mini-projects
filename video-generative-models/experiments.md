# Prompt Relay on CogVideoX1.5-5B: results so far

Oct 14, 2026 · @Shibe

Configuration `block` (one stock prompt per event, "global + event", with per-frame normalisation and each block's text queries restricted to that block's text keys) gives sharp video with no haze or TV static in all 9 runs across four scenes, but the first event is weak in the first 1–2 s in 5 of them. Configuration `global_local` (round 4), which adds a standalone global prompt that every frame sees, makes the opening subject clear in all 4 gem and child runs and stays sharp, but later events are weaker and in one run the boy disappears at 12–14 s. In round 5, a fixed global share of 0.5 changed nothing visible; a share of 0.25 kept every opening clear and improved later events in three of five runs, but in both child seeds the subject's appearance changed at 7–9 s. Normalisation is needed to keep `global_local` free of haze, and text queries must see only their own block's text keys or the events never change. All results are from visual inspection.

## Changes to Prompt Relay's method

The current method (`--relay_text_encoding block`) is configuration block in "Configurations tested". It keeps the paper's penalty formula and changes how it is applied in eight places. The paper ([arXiv 2604.10030](https://arxiv.org/abs/2604.10030)) adds this penalty to the cross-attention logits of Wan2.2-T2V-A14B, with ε = 0.1, w = L − 2 and f(i) the latent frame of query i:

```latex
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d} - C\right)V,\qquad C(i,j)=\frac{\mathrm{ReLU}(|f(i)-m_s|-w)^2}{2\sigma^2},\qquad \sigma=\frac{L-w}{\sqrt{2\ln(1/\epsilon)}}
```

1. **Where the penalty sits.** CogVideoX1.5-5B concatenates text and video tokens and runs one attention with shared Q/K/V projections, so a video query's softmax covers text keys and video keys together. The penalty is added only to the logits of video queries on event text keys; video queries on video keys get no bias. In the paper's cross-attention, video queries attend to text keys only, so the softmax covers text keys only.
2. **Time unit.** CogVideoX1.5 packs 2 latent frames into each token (patch\_size\_t = 2), so distances are measured in temporal patches of 0.5 s and w = L − 1 patch (the paper's 2 latent frames). Times are mapped on the padded latent grid, whose first latent frame is discarded after denoising, so the schedule lands 0.25 s earlier than the JSON times (unchanged from v0).
3. **Event prompts.** Each event's text is one complete prompt, "global + event k", encoded by T5 on its own as a stock 226-token prompt (content, end token, padding). For video queries, the penalty covers all of block k's keys, including its copy of the global prompt, so there is no separate global prompt that every frame always sees. The paper uses a global prompt for the whole video and does not say how prompts are encoded.
4. **Normalisation across events (added).** For each video-query frame, the biases on the event blocks' keys are shifted by their log-sum-exp, so the events' weights e^(−C) sum to 1. For the `child_lowtoken` timings, the two adjacent events keep 0.10 each at the 3.0 s and 6.0 s boundaries without it (0.20 total) and 0.50 each with it; away from the boundaries the weights are the same to two decimals. The paper has no normalisation.
5. **Text-query mask (added).** CogVideoX text tokens are also queries. Text queries of block k attend only to the text keys of block k (other blocks blocked) and to all video keys with no bias. In the paper's cross-attention, text tokens are only keys and values, never queries.
6. **Negative (CFG) branch.** The penalty is applied only to the conditional branch. The negative branch is the stock single 226-token empty prompt; all its queries attend to all its keys with no bias. The paper does not say how it handles this branch.
7. **Unbounded outer edges.** The first event has no penalty before its midpoint and the last event none after its midpoint. This came from the v0 code; the paper does not describe it.
8. **Layers and steps.** The penalty is applied in all 42 transformer layers at all 50 denoising steps. The paper does not say which layers or steps it uses.

## Configurations tested

Every configuration is described in full on its own below; the runs in "Experiments and results" name the configuration they used.

**Terms**

- **Query and key.** In every attention layer, each text token and each video token is a query, and the keys are all text and video tokens of the same CFG branch (and, with the sliding window, of the same window). "Video queries → text keys" means the attention scores of video tokens over text tokens; the other three directions are named the same way.
- **Bias.** A number added to an attention logit after the 1/√d scaling. "No bias" = untouched (stock). "Blocked" = −inf (zero attention). "Penalty" = −C\_k(f) for a key in event k's tokens.
- **Penalty C\_k(f).** C\_k(f) = ReLU(|f − m\_k| − w\_k)² / (2σ\_k²), with f the temporal patch of the video token (0.5 s, 2 latent frames), m\_k the event's midpoint, w\_k its half-length minus 1 patch, σ\_k = (half-length − w\_k) / √(2 ln 10) ≈ 0.466. The weight multiplier e^(−C) is 1 inside the event, 0.1 at its boundary and ≈ 0 beyond. The first event has no penalty before its midpoint, the last event none after its midpoint. With the sliding window, f is absolute time.
- **Normalisation (lse).** For each video-query frame f, the event biases are shifted so their weights e^(bias) sum to 1: bias\_k = −C\_k(f) − log Σ\_j e^(−C\_j(f)). "None" = the raw −C\_k(f).
- **Stock 226-token prompt.** One string tokenised, padded to 226 tokens (content, end token, padding) and encoded by T5 alone, with no T5 attention mask.
- **Negative branch.** The unconditional CFG branch, prompt "" (empty).

**Common to every configuration:** CogVideoX1.5-5B, 50 steps, guidance 6 with stock dynamic guidance; where a penalty is used, it is applied in all 42 layers at every step. Video queries → video keys never get a bias. Text tokens have no positional embedding; the rotary embedding is applied only to video queries and video keys. The JSON-to-frame mapping is the v0 mapping in every run so far (0.25 s early).

| Configuration | Runs | Result |
| --- | --- | --- |
| `stock` | A0 (seeds 42, 7), S | sharp; later events missing |
| `v0` | v0 baseline (10 s, 15 s) | hazy |
| `v0 attempt 1` | attempt 1 | TV static |
| `v0 attempt 2` | attempt 2 | TV static |
| `joint, no penalty` | A | frame-identical to A0 |
| `joint + penalty` | B | TV static |
| `split + penalty` | C, E | hazy |
| `split, no penalty` | C0 | very hazy |
| `split + penalty + quarantine` | D | hazier, particles |
| `block` (current method) | F (seeds 42, 7), E2, all round 3 relay runs | sharp; first event often weak at 0–2 s |
| `block + text-query time penalty` | F2 | TV static |
| `blend` (not Prompt Relay) | G | sharp |
| `global_local` | round 4 G1: gem (seeds 42, 7), child (seeds 42, 7), eagle (seed 42) | sharp; opening clear in all gem and child runs; later events weaker than `block`; boy gone at 12–14 s in child seed 42; no eagle |
| `global_local, no normalisation` | round 4 G1\_none: child (seed 42) | extreme haze; boy dissolves by 7–8 s |
| `block, text queries see all text` | round 4 M\_ttfull: gem (seed 42) | sharp; empty opening; never leaves event 1 |
| `global_local, global share 0.5` | round 5 G3: child (seed 42), gem (seed 42) | nearly frame-for-frame the same as `global_local` |
| `global_local, global share 0.25` | round 5 G4: gem (seeds 42, 7), child (seeds 42, 7), eagle (seed 42) | sharp; opening clear in all gem and child runs; all five gem shapes right in seed 7; subject or bedding changes at 7–9 s in both child seeds; airplane never flown; no eagle |

### stock

- **Text (conditional branch):** one stock 226-token prompt of the global prompt and all event prompts joined by spaces. Text beyond 226 tokens is cut (in S, event 5's sentence was cut).
- **Penalty:** none.
- **Video queries → text keys:** no bias.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** no bias.
- **Text queries → video keys:** no bias.
- **Normalisation:** not applicable.
- **Negative branch:** stock 226-token "", no bias anywhere.

### v0

- **Text (conditional branch):** the global prompt and each event prompt encoded by T5 separately, each padded to 224 tokens, then concatenated: \[global 224\] \[event 1 224\] \[event 2 224\] …
- **Penalised tokens:** each event's whole 224-token block (content, end token, padding). The global block is never penalised.
- **Video queries → text keys:** −C\_k(f) on event k's block; no bias on the global block.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** global-block queries see only global-block keys (all event blocks blocked). Event-k queries see event-k keys and global-block keys (other event blocks blocked).
- **Text queries → video keys:** event-k queries get −C\_k(f) on video keys at frame f. Global-block queries: no bias.
- **Normalisation:** none.
- **Negative branch:** the same number of blocks, each "" padded to 224 and stacked. Every bias and block above is applied to it identically, by position.

### v0 attempt 1

Described from your account and the encoding snippet; the rest of its code was not reviewed.

- **Text (conditional branch):** global and event prompts each encoded by T5 separately with no padding and no truncation, then concatenated.
- **Negative branch:** one "" block per prompt block, padded or cut to that block's length; no mask or penalty.
- **Penalty:** applied to the conditional branch only.

### v0 attempt 2

Described from the attention snippet; its text encoding was not shown.

- **Video queries → text keys (conditional branch):** −C\_k(f) on event k's tokens.
- **Video queries → video keys:** no bias.
- **Text queries → text keys and text queries → video keys (conditional branch):** no bias.
- **Normalisation:** none.
- **Negative branch:** no bias anywhere.

### joint, no penalty

- **Text (conditional branch):** one stock 226-token prompt of the global prompt and all event prompts joined by spaces (the same embedding `stock` uses). Each event's token span inside it is located but not used.
- **All four directions:** no bias.
- **Negative branch:** stock 226-token "", no bias.

### joint + penalty

- **Text (conditional branch):** one stock 226-token prompt of the global prompt and all event prompts joined by spaces; each event's content-token span inside it is located.
- **Penalised tokens:** each event's content tokens inside the joined prompt. The global prompt's tokens, the end token and the padding are never penalised.
- **Video queries → text keys:** −C\_k(f) on event k's tokens; no bias on all other text keys.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** no bias.
- **Text queries → video keys:** no bias.
- **Normalisation:** none.
- **Negative branch:** stock 226-token "", no bias.

### split + penalty

- **Text (conditional branch):** \[global prompt as a stock 226-token prompt\] followed by each event's own tokens. Each event is encoded by T5 alone (padded inside T5) and only its content tokens and end token are kept.
- **Penalised tokens:** each event's kept tokens. The global block (226) is never penalised.
- **Video queries → text keys:** −C\_k(f) on event k's tokens; no bias on the global block.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** no bias.
- **Text queries → video keys:** no bias.
- **Normalisation:** none.
- **Negative branch:** stock 226-token "", no bias.

### split, no penalty

- **Text (conditional branch):** \[global prompt as a stock 226-token prompt\] followed by each event's own tokens. Each event is encoded by T5 alone (padded inside T5) and only its content tokens and end token are kept.
- **All four directions:** no bias.
- **Negative branch:** stock 226-token "", no bias.

### split + penalty + quarantine

- **Text (conditional branch):** \[global prompt as a stock 226-token prompt\] followed by each event's own tokens. Each event is encoded by T5 alone (padded inside T5) and only its content tokens and end token are kept.
- **Penalised tokens:** each event's kept tokens; the global block is never penalised.
- **Video queries → text keys:** −C\_k(f) on event k's tokens; no bias on the global block.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** global-block queries see only global-block keys. Event-k queries see event-k keys and global-block keys (other events blocked).
- **Text queries → video keys:** event-k queries get −C\_k(f) on video keys at frame f. Global-block queries: no bias.
- **Normalisation:** none.
- **Negative branch:** stock 226-token "", no bias.

### block (current method)

- **Text (conditional branch):** one stock 226-token prompt per event, "global + event k", concatenated: \[block 1\] \[block 2\] … (K × 226 tokens). There is no standalone global block.
- **Penalised tokens:** each whole block: its copy of the global prompt, the event text, the end token and the padding.
- **Video queries → text keys:** normalised bias on block k's keys.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** block-k queries see only block-k keys (other blocks blocked).
- **Text queries → video keys:** no bias.
- **Normalisation:** lse.
- **Negative branch:** stock 226-token "", no bias.

### block + text-query time penalty

- **Text (conditional branch):** one stock 226-token prompt per event, "global + event k", concatenated: \[block 1\] \[block 2\] … (K × 226 tokens). There is no standalone global block.
- **Penalised tokens:** each whole block.
- **Video queries → text keys:** normalised bias on block k's keys.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** block-k queries see only block-k keys.
- **Text queries → video keys:** block-k queries get block k's normalised bias on video keys at frame f.
- **Normalisation:** lse.
- **Negative branch:** stock 226-token "", no bias.

### blend (not Prompt Relay)

- **Text:** the same K "global + event k" stock 226-token prompts as `block`, each used as the prompt of its own pass.
- **Each denoising step:** one stock pass per event prompt and one stock pass with "" for the negative branch. In every pass all four directions have no bias.
- **Conditional prediction:** at each latent frame, the event passes' predictions are mixed with weights e^(bias\_k(f)) from the normalised penalty (they sum to 1).

### global\_local (round 4)

- **Text (conditional branch):** \[global prompt as a stock 226-token prompt\] \[event 1 alone as a stock 226-token prompt\] \[event 2 alone\] …
- **Penalised tokens:** each whole event block. The global block is never penalised; it is treated as an event with an infinite plateau (C = 0 at every frame).
- **Video queries → text keys:** normalised bias over all blocks including the global block. Weights inside event k: global 0.5, event k 0.5. At a 3 s boundary: global ≈ 0.83, each adjacent event ≈ 0.08.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** each block's queries see only its own block's keys (global sees global, event k sees event k).
- **Text queries → video keys:** no bias.
- **Normalisation:** lse.
- **Negative branch:** stock 226-token "", no bias.

### global\_local, no normalisation (round 4)

- **Text (conditional branch):** \[global prompt as a stock 226-token prompt\] \[event 1 alone as a stock 226-token prompt\] \[event 2 alone\] …
- **Video queries → text keys:** no bias on the global block; −C\_k(f) on event k's block. Inside an event, the global and that event both have weight 1.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** each block's queries see only its own block's keys.
- **Text queries → video keys:** no bias.
- **Normalisation:** none.
- **Negative branch:** stock 226-token "", no bias.

### block, text queries see all text (round 4)

- **Text (conditional branch):** one stock 226-token prompt per event, "global + event k", concatenated: \[block 1\] \[block 2\] … (K × 226 tokens). There is no standalone global block.
- **Video queries → text keys:** normalised bias on block k's keys.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** no bias (every block's queries see all blocks' keys).
- **Text queries → video keys:** no bias.
- **Normalisation:** lse.
- **Negative branch:** stock 226-token "", no bias.

### global\_local, global share 0.5 (round 5)

- **Text (conditional branch):** \[global prompt as a stock 226-token prompt\] \[event 1 alone as a stock 226-token prompt\] \[event 2 alone\] …
- **Penalised tokens:** each whole event block. The global block is never penalised.
- **Video queries → text keys:** a fixed bias of log 0.5 on every key of the global block, at every frame. Event block k gets −C\_k(f), normalised over the event blocks only, then scaled so the event blocks together get 0.5. Weights inside event k: global 0.5, event k 0.5. At a 3 s boundary: global 0.5, each adjacent event 0.25.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** each block's queries see only its own block's keys (global sees global, event k sees event k).
- **Text queries → video keys:** no bias.
- **Normalisation:** lse over the event blocks; the global block's weight is fixed at 0.5.
- **Negative branch:** stock 226-token "", no bias.

### global\_local, global share 0.25 (round 5)

- **Text (conditional branch):** \[global prompt as a stock 226-token prompt\] \[event 1 alone as a stock 226-token prompt\] \[event 2 alone\] …
- **Penalised tokens:** each whole event block. The global block is never penalised.
- **Video queries → text keys:** a fixed bias of log 0.25 on every key of the global block, at every frame. Event block k gets −C\_k(f), normalised over the event blocks only, then scaled so the event blocks together get 0.75. Weights inside event k: global 0.25, event k 0.75. At a 3 s boundary: global 0.25, each adjacent event 0.375.
- **Video queries → video keys:** no bias.
- **Text queries → text keys:** each block's queries see only its own block's keys (global sees global, event k sees event k).
- **Text queries → video keys:** no bias.
- **Normalisation:** lse over the event blocks; the global block's weight is fixed at 0.25.
- **Negative branch:** stock 226-token "", no bias.

## Experiments and results

Every run used CogVideoX1.5-5B text-to-video at 1360 × 768, 16 fps, 50 steps, guidance 6 with the stock dynamic guidance, and seed 42 unless noted. In rounds 1 and 2, 10 s runs are 161 frames of `child_lowtoken` (events at 0–3, 3–6 and 6–10 s); 15 s runs are 241 frames of `child` (five 3 s events) with the sliding window. Round 3 adds 15 s runs of gem (five 3 s events), table (five overlapping events: 0–5, 3–8, 6–10, 8–13, 11–15 s) and eagle (events at 0–3, 3–7, 7–11, 11–15 s). The table scene is dropped from later rounds. Each run names its configuration from "Configurations tested".

1. **v0 baseline · `v0`.** 10 s (`child_lowtoken`) and 15 s (`child`, sliding window). Result: no static; hazy and blurry at 10 s and 15 s; the boy morphs. At 15 s the boy stayed in frame for all 15 s.
2. **v0 attempt 1 · `v0 attempt 1`.** Result: TV static.
3. **v0 attempt 2 · `v0 attempt 2`.** Result: TV static.
4. **Round 1 · A0 · `stock`.** Result: sharp, detailed room; the boy morphs.
5. **Round 1 · A · `joint, no penalty`.** Result: frame-identical to A0.
6. **Round 1 · B · `joint + penalty`.** Result: TV static.
7. **Round 1 · C · `split + penalty`.** Result: no static; hazy, less detail than A0; the boy jumps, then picks up and plays with a morphing object.
8. **Round 1 · D · `split + penalty + quarantine`.** Result: hazier than C; the boy breaks into particle-like artifacts.
9. **Round 1 · E · `split + penalty`, 15 s, sliding window.** Result: hazy; the boy is lying on the bed at 1 s, walks out of frame at \~6 s, and a different boy in different clothes appears on the bed at \~10 s.
10. **Round 2 · C0 · `split, no penalty`.** Result: extremely hazy; no window, furniture or toys rendered; the boy lies face-down at 0–2 s.
11. **Round 2 · F · `block`, seeds 42 and 7.** Result: sharp and detailed in both seeds; seed 7 matches `stock` A0 seed 7 in clarity and lighting. Opening: seed 42 shows an empty bed at 0 s and the head pops up at 1–2 s; seed 7 shows only legs and hip at the left edge at 0–2 s. In both seeds the boy holds and waves a toy from \~6 s (event 3). Event 2 (stand up, jump repeatedly) is not performed in either seed. Limbs and toys morph in fast motion.
12. **Round 2 · F2 · `block + text-query time penalty`.** Result: TV static.
13. **Round 2 · G · `blend` (not Prompt Relay).** Result: quality comparable to F seed 42, slightly crisper textures. Event 1 shown (boy lying on the bed at 0–2 s); event 2 partly (hands and knees, no jumping); event 3 partly (stands at \~6 s and grabs a red object).
14. **Round 2 · A0 seed 7 · `stock`.** Result: sharp, no haze; empty bed at 0–1 s, the boy slides in at 2 s; event 3 never occurs; severe limb morphing during the jumps.
15. **Round 2 · E2 · `block`, 15 s, sliding window.** Result: sharp; only a lump under the blanket at 0–2 s; the same boy in the same clothes is in frame from 3 s to 14 s. Events 3 and 4 happen in their windows (airplane flown at \~6–8 s; thrown at \~9 s, lands on the floor at \~10–11 s). Event 2's jumping is not performed; in event 5 he walks right instead of to the airplane.
16. **Round 3 · S · `stock`, 15 s, sliding window, child.** The joined prompt exceeded 226 tokens, so event 5's sentence was cut off. Result: the boy is visible lying on the bed at 0–2 s, rolls and stands at 3–5 s; at \~6–7 s a second boy appears in the middle of the bed and the first leaves the frame; events 3–5 never happen.
17. **Round 3 · `block`, 15 s, sliding window, child seed 7.** Result: sharp; only a blanket lump at 0–1 s, the boy's head emerges at 2–3 s; he stands at 4 s and leaps at 5–6 s (event 2); he holds and swoops a yellow toy airplane at 8–11 s; he never throws it or picks it up (events 4–5).
18. **Round 3 · `block`, 15 s, sliding window, gem seed 7.** Result: sharp; a small but visible purple gem in the middle from 0 s that changes shape (not a diamond) and becomes a much larger gold star at 3 s; star at 3–5 s; flat red wedge at 6–8 s; green jade slab at 9–11 s; blue sphere at 12–14 s.
19. **Round 3 · `block`, 15 s, sliding window, gem seed 42.** Result: sharp; empty studio at 0–1 s and a thin purple sliver at 2 s; purple shows at 2–3 s while the star forms; the star, red pyramid, green cube and blue sphere each appear in their own 3 s windows.
20. **Round 3 · `block`, 15 s, sliding window, table seed 7.** Result: sharp; the purple gem drops onto the table at 0–2 s; later objects mostly morph out of the previous object instead of dropping and rolling off; one object sinks through the tabletop and the table legs change shape.
21. **Round 3 · `block`, 15 s, sliding window, table seed 42.** Result: sharp; the purple gem starts on the floor and slides instead of dropping; later objects morph from one into the next near the table and the table changes shape at 4 s; none rolls off an edge.
22. **Round 3 · `block`, 15 s, sliding window, eagle seed 42.** Result: sharp; no eagle, as the video opens inside a circular iris with the city already in it; city street at 3–7 s; the man in sunglasses in the car at \~9 s; a vintage TV in a living room from 10 s, with no people.
23. **Round 4 · M\_ttfull · `block, text queries see all text`, 15 s, sliding window, gem seed 42.** Result: no haze; empty studio at 0–1 s and a thin purple sliver at 2 s; from 3 s to 14 s one dark-purple faceted gem that never turns into the star, pyramid, cube or sphere.
24. **Round 4 · G1\_none · `global_local, no normalisation`, 15 s, sliding window, child seed 42.** Result: extreme peach/orange haze and light bloom for all 15 s, with no furniture, walls or toys visible. A faint silhouette of the boy lies on the mattress at 0–2 s, pushes up and kneels at 3–6 s, and dissolves into the haze by 7–8 s; empty mattress to the end.
25. **Round 4 · G1 · `global_local`, 15 s, sliding window, gem seed 42.** Result: sharp, no haze; a large, clear deep-purple diamond from 0 s (a dark purple/amber block at 2 s); gold star at 3–4 s and a pink transitional block at 5 s; a red block with a flat top instead of a pointed pyramid at 6–8 s; green jade cube at 9–11 s with blue edges at 11 s; sapphire orb at 12–14 s. Shapes are less precise than in `block` gem seed 42.
26. **Round 4 · G1 · `global_local`, 15 s, sliding window, gem seed 7.** Result: sharp, no haze; a large, clear deep-purple diamond at 0–2 s; a gold-and-silver star/octahedron at 3–4 s and an amber square at 5 s; a square that turns magenta then red at 6–8 s (no pyramid); a green jade block at 9–10 s that splits to show blue at 11 s; sapphire sphere at 12–14 s.
27. **Round 4 · G1 · `global_local`, 15 s, sliding window, child seed 42.** Result: sharp, no haze; the boy clearly visible lying on the mattress at 0–2 s; hands and knees at 3–4 s, stands or kneels at 5 s; crawls toward the toys at 6–9 s with no airplane; rolls onto his back at 10 s, shrinks into a small grey bundle at 11 s, and is gone from the room at 12–14 s.
28. **Round 4 · G1 · `global_local`, 15 s, sliding window, child seed 7.** Result: sharp, no haze; the boy clearly visible from 0 s, sitting upright (not lying); stands and jumps at 3–5 s (event 2); sits and plays with yellow toys on the bed at 6–12 s, never flying or throwing the airplane; stands next to the toys at 13–14 s.
29. **Round 4 · G1 · `global_local`, 15 s, sliding window, eagle seed 42.** Result: no haze; textures more painterly than `block`; no eagle (opens on a ring of clouds around a dark silhouette and the city); city street at 3–7 s; a distorted car at 8–9 s with no man in sunglasses; a CRT TV showing a warm room inside its screen at 11–14 s, while the background stays the city.
30. **Round 5 · G3 · `global_local, global share 0.5`, 15 s, sliding window, child seed 42.** Result: nearly frame-for-frame the same as G1 child seed 42. The blond boy lies on the mattress at 0–2 s, gets onto hands and knees at 3–4 s and kneels at 5 s, crosses the bed into the window light at 6–9 s with no airplane, rolls onto his back at 10 s, shrinks into a small dark bundle at 11 s, and is gone at 12–14 s.
31. **Round 5 · G3 · `global_local, global share 0.5`, 15 s, sliding window, gem seed 42.** Result: nearly the same as G1 gem seed 42. Deep-purple diamond at 0–1 s (a dark purple/brown block at 2 s); amber-gold star at 3–4 s and a salmon-pink block at 5 s; red block with a flat top at 6–8 s; green jade block at 9–11 s with a blue top edge at 11 s; blue sphere at 12–14 s (green patch at 12 s).
32. **Round 5 · G4 · `global_local, global share 0.25`, 15 s, sliding window, child seed 42.** Result: sharp, no haze. A curly-haired boy in a light-blue hoodie lies on his stomach on the bed at 0–2 s, pushes up at 3–4 s, stands at 5 s and jumps at 6 s. At 7–8 s he blurs into a silhouette; from 9 s he is a different blond boy in an orange shirt. He sits with a large toy airplane at 9–12 s (never flown or thrown) and walks right at 13–14 s; he stays in frame to the end.
33. **Round 5 · G4 · `global_local, global share 0.25`, 15 s, sliding window, child seed 7.** Result: sharp, no haze. A boy in a striped T-shirt kneels upright on the bed at 0–2 s (not lying) and jumps repeatedly at 3–6 s. At 7–8 s the sheets billow, the bedding becomes a blue-and-white striped mat with a shelf, and his shirt turns solid grey. He kneels holding a small toy at 9–12 s (no airplane flown or thrown) and walks right toward the toys at 13–14 s.
34. **Round 5 · G4 · `global_local, global share 0.25`, 15 s, sliding window, gem seed 42.** Result: sharp, no haze. Deep-purple diamond at 0–2 s; gold star at 3–4 s (four points at 3 s, five at 4 s) and a pink/red block at 5 s; red block with a flat top (truncated pyramid) at 6–8 s; green jade block at 9–10 s with a blue dome rising from it at 11 s; blue sphere at 12–14 s (green patch at 12 s).
35. **Round 5 · G4 · `global_local, global share 0.25`, 15 s, sliding window, gem seed 7.** Result: sharp, no haze. Large purple diamond at 0–2 s; gold five-pointed star at 3–4 s (four points at 5 s); translucent red triangular pyramid at 6–8 s; textured green jade cube at 9–11 s; blue sphere at 12–14 s. Every shape matches its event.
36. **Round 5 · G4 · `global_local, global share 0.25`, 15 s, sliding window, eagle seed 42.** Result: sharp. No eagle: opens inside a white cloud/iris ring with the city in the centre and zooms through it by 2 s; cyberpunk street with traffic at 3–7 s; a car roof at 8 s and a car's side window at 9 s with a pink blur inside, no man in sunglasses; a vintage TV at 10 s, then a living room with lamps and a sofa inside its screen and a warm room with a lamp and a standing figure around it at 11–14 s.

## Sliding window and text length

The sliding window is unchanged from v0. The text-length method changed from stacked padded prompts (v0) to one "global + event" block per event.

1. **Window settings (same in v0, E and E2).** Windows of 42 latent frames with a stride of 20, so a 241-frame video gets two windows. Each window gets its own rotary position embedding with temporal positions 0–20, the same as a native 161-frame generation. Each window is a separate transformer pass, so video queries attend only to the video keys of their own window. Noise predictions are blended linearly over the 22 shared latent frames before each scheduler step. Each window passes its start offset to Prompt Relay, so event timing uses absolute time.
2. **What each window covers.** Window 1 covers 0.0–10.0 s of the output and window 2 about 4.6–15.0 s, so they overlap for about 4.6–10.0 s. The latent sequence is padded from 61 to 62 frames and the first latent frame is dropped after denoising.
3. **E vs E2.** Same windows and seed; only the configuration differed (split + penalty vs block).
4. **Per-event budget.** Each event block is one standard T5 prompt of 226 tokens, so each event's budget is 225 tokens minus the global prompt's tokens. With K events the conditional branch carries K × 226 text tokens; the negative branch carries 226, zero-padded to the same length for batching and stripped again before attention.
5. **Why blocks can be concatenated.** CogVideoX1.5-5B adds no positional embedding to text tokens: its config sets `use_rotary_positional_embeddings` true and `use_learned_positional_embeddings` false, and the rotary embedding is applied only to video queries and video keys. v0 already ran with up to 1,344 text tokens.
6. **Prompt length setting.** v0 used `max_sequence_length` 224, the model card's prompt token limit. Every run since round 1, including stock A0, uses 226, the diffusers default and the transformer config's `max_text_seq_length`.

## What the runs show and what is not yet tested

Each finding comes from one run per configuration unless a second seed is noted. Configuration names refer to "Configurations tested".

**Shown**

- `joint, no penalty` → `joint + penalty` (A → B): adding only the penalty on video queries → event text keys produced TV static.
- `split, no penalty` (C0) was extremely hazy, so the haze of `split + penalty` (C) did not come from the penalty.
- `split + penalty` → `split + penalty + quarantine` (C → D): adding the text queries → text keys mask and the penalty on text queries → video keys made the video hazier.
- `block` → `block + text-query time penalty` (F → F2): adding the penalty on text queries → video keys produced TV static.
- `block` → `block, text queries see all text` (round 3 gem seed 42 → M\_ttfull): the video never left event 1 (one purple gem from 3 s to 14 s), and the opening was still empty. In `block`, text queries seeing only their own block's text keys is needed for the events to change, and it is not what causes the weak opening.
- `global_local` → `global_local, no normalisation` (child seed 42, G1 → G1\_none): without normalisation the video was extremely hazy and the boy dissolved by 7–8 s; with it, sharp. The other layouts where video queries saw a global block and event text all at full weight, `v0` and `split, no penalty`, were also hazy.
- `block` → `global_local` (same scenes and seeds): the opening subject was clearly visible from 0 s in all four gem and child runs (in `block` it was missing, hidden or small in all four). Later events were weaker: child seed 42 never flew or threw the airplane and the boy was gone at 12–14 s (E2 kept him, flew and threw it); child seed 7 never flew the airplane (round 3 `block` seed 7 did); eagle lost the man in sunglasses and the living room; gem seed 42 shapes were less precise. Neither layout showed the eagle.
- Computed text weights for video queries (e^bias). `block`: inside event k, block k ("global + event k") gets 1.0; at a 3 s boundary, the two adjacent blocks get 0.5 each. `global_local`: inside event k, the global block 0.5 and event k 0.5; at a 3 s boundary, global 0.83 and each adjacent event 0.08.
- `block` produced no haze in any of its 9 runs (child, gem, table, eagle; seeds 42 and 7).
- At 15 s with the sliding window, `block` kept one boy in both child seeds (E2 seed 42, round 3 seed 7). With seed 42, `stock` (S) produced a second boy in the window overlap, and `split + penalty` (E) lost the boy.
- Opening 1–2 s of `block` runs (the subject is usually partly there: a limb, a small lump, a small gem, or a purple tint at 2–3 s): absent or hidden in F seed 42, E2, child seed 7 and gem seed 42; replaced by later content in eagle; small or partial in F seed 7 and gem seed 7; present in both table runs.
- The opening problem also occurs without Prompt Relay: `stock` A0 seed 7 starts with an empty bed. With the same seed as a failing `block` run, the boy was visible at the start in `stock` S (seed 42, vs E2) and in `blend` G (seed 42, vs F).
- Computed from the code's formulas: in `block`, normalisation changes event 1's bias for video queries by at most 0.0001 over 0–2.5 s, and the 0.25 s schedule shift only moves the boundary half-second (2.5–3.0 s). Neither changes what the first 2.5 s are conditioned on.
- In gem seed 42 (`block`), each later event (star, pyramid, cube, sphere) appeared in its own 3 s window.
- In both table seeds (`block`), later objects morph out of the previous object instead of dropping and rolling off the table.
- `blend` (G) gave quality comparable to `block` (F) in seed 42.
- Human morphing in fast motion also happens in `stock` (A0, seeds 42 and 7).
- `global_local` → `global_local, global share 0.5` (G1 → G3, child seed 42 and gem seed 42): only the weights near event boundaries changed (each adjacent event 0.08 → 0.25, global 0.83 → 0.5). Both videos were nearly frame-for-frame the same as G1, so the boundary weights are not what weakens later events in `global_local`.
- `global_local` → `global_local, global share 0.25` (G1 → G4, same five scenes and seeds): inside every event the weights changed from global 0.5 / event 0.5 to 0.25 / 0.75. Openings stayed clear in all four gem and child runs. Gem seed 7 showed all five shapes, including a pointed pyramid (G1 had none); eagle regained the living room around the TV; in child seed 42 the boy jumped and stayed to 14 s (G1: gone at 12–14 s). Unchanged: no eagle, no man in sunglasses, gem seed 42's pyramid still flat-topped.
- In both child seeds, `global_local, global share 0.25` changed the subject at 7–9 s, inside the window overlap (about 4.6–10.0 s): seed 42 became a different boy, seed 7 changed shirt and bedding. With the same seeds, `global_local` (G1) and `block` (E2, round 3 seed 7) kept one appearance.
- Child events 3–4 (fly, then throw the airplane) happened only in `block` runs (E2 did both; round 3 seed 7 flew it). They never happened in the five child runs of the `global_local` family (G1 × 2, G3, G4 × 2).
- The eagle never appeared in any eagle run (`block`, `global_local`, `global_local, global share 0.25`); all three open inside a ring with the city already in it.

**Not yet tested**

- `block` without normalisation: never run.
- `global_local` with a global share between 0.25 and 0.5: never run.
- Why `block` weakens the opening and `global_local` weakens later events, and why a global share of 0.25 changes the subject in the window overlap, have not been established.
- Why the failed configurations produce haze or static has not been established.
- No quantitative metric was computed; every result is visual inspection.
- Most configurations have one seed per scene; `block` has two on child (10 s and 15 s), gem and table; `global_local` has two on child and gem, and so does its 0.25 global share version.

## Sources

- [Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation](https://arxiv.org/abs/2604.10030) (Chen, Huang, Liu; arXiv 2604.10030): base model, penalty formula, ε, w
- [CogVideoX1.5-5B model card](https://huggingface.co/THUDM/CogVideoX1.5-5B/blob/main/README.md): 224-token prompt limit, 16 fps, 3D RoPE
- [CogVideoX1.5-5B transformer config](https://huggingface.co/THUDM/CogVideoX1.5-5B/blob/main/transformer/config.json): 42 layers, patch\_size\_t 2, max\_text\_seq\_length 226, positional-embedding flags
