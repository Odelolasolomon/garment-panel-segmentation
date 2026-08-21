# DESIGN_NOTE.md — Shading-Aware Fabric Compositing

## The problem with flat tiling in production

Part 2's flat tile works because it never has to reconcile with anything underneath
it — it just paints over the panel boundary. In production the garment image is a
shaded 3D render: the surface already has folds, drape, ambient occlusion, and a
directional light source baked into its pixel values. A flat tile ignores all of
that and sits on top as a uniform, unshaded layer. The eye immediately reads this as
wrong, for the same reason a pasted-on decal reads as a decal rather than paint: real
fabric brightens and darkens with the surface it's wrapped around, and a flat tile
does neither.

## Approach: separate luminance from chromaticity, recombine

The core idea is to stop treating this as "paste texture into region" and start
treating it as "transplant the *color/pattern* of the fabric while keeping the
*existing shading* of the render." Concretely:

1. **Extract a shading/luminance signal from the original render**, restricted to
   the panel region. If the renderer exposes a separate shading pass (an ambient
   occlusion pass, a diffuse-only pass, or ideally a normal map) that is the cleanest
   source, because it isolates lighting information from the base material color
   entirely. If only the final flattened RGB render is available, a reasonable
   approximation is converting that region to grayscale/luminance and treating it as
   a proxy shading map — noisier, but usable, and often the difference between "does
   this look plausible at a glance" and "would this withstand close inspection" is
   small enough not to justify the extra rendering-pipeline dependency at first pass.

2. **Recombine the fabric's own color/pattern with that extracted luminance**, rather
   than pasting the fabric at 100% opacity. The standard way to do this is a
   multiply-style blend: convert the fabric tile to the same color space, multiply
   its channels by the normalized luminance map from step 1 (or use a more
   photometrically correct transfer such as applying the shading map's luminance
   channel directly to the fabric in an HSL/Lab space while keeping the fabric's own
   hue and saturation). The effect is that wherever the original render was dark
   (a fold's shadow, the underside of a wrinkle), the composited fabric darkens to
   match; wherever it was lit, the fabric brightens to match. This is what makes it
   read as "fabric wrapped around a shape" instead of "sticker."

3. **If UV-mapped geometry is available**, go one step further and warp the fabric
   texture to follow the panel's actual 3D curvature (project the flat swatch
   through the mesh's UV coordinates before the luminance blend) so the fabric's own
   pattern — stripes, a grid, a print — bends and compresses the way real cloth would
   across a fold, rather than staying rectilinear while only its brightness changes.
   This is the difference between "correctly lit flat tile" and genuine drape.

## What this needs that a flat tile does not

- **A shading/luminance source separate from the final RGB.** Best case: the render
  pipeline exposes a diffuse/shading pass or a normal map per image. Worst case: we
  derive an approximate luminance map from the flattened render itself, which is
  strictly worse (any color information already baked into the "original" fabric
  bleeds into what should be a pure lighting signal) but requires no changes to the
  rendering pipeline, which matters if we don't control it.
- **UV coordinates per panel**, only if attempting true drape-following warping
  (step 3). This is a meaningfully bigger dependency — it requires the 3D asset
  pipeline to export UV data alongside the flattened render, not just a rendered
  image. Without it we can still get correctly-*shaded* fabric, just not
  correctly-*warped* fabric — the pattern itself won't bend with the fold, only its
  brightness will.
- **No new trained model, if using the luminance-transfer approach.** This is
  deliberately still not a machine-learning problem — it's a compositing operation
  operating on whatever shading information is available, consistent with the
  philosophy set up in Part 1/Part 2 that "put fabric in the right place" and "make
  fabric look right" are both engineering problems, not generative ones, as long as
  the render pipeline gives us the inputs to work with.

## Where it breaks down

- **Approximate luminance from flattened RGB (no separate shading pass) will fight
  with strongly colored or patterned original fabric.** If the placeholder garment
  fabric before compositing was itself brightly colored, that color contaminates the
  "luminance" signal we extract, and the new fabric inherits tinting it shouldn't
  have. This is the most likely first visible failure mode and the strongest
  argument for getting an actual shading-only pass from the renderer rather than
  approximating from final RGB.
- **Hard occlusion boundaries (a panel partly hidden behind another piece, a strap,
  a fold that fully self-occludes) aren't solved by luminance transfer at all** —
  that's a masking problem, already handled upstream by Part 1's segmentation, but
  errors there (a panel boundary that's slightly wrong) become directly visible once
  the fill actually looks like real fabric instead of a flat color that's easy to
  visually forgive.
- **Without UV warping, patterned fabrics (stripes, checks, logos) will look shaded
  correctly but geometrically flat** — a striped fabric across a sleeve that bends
  will have realistic light/dark shading along the bend but the stripes themselves
  will stay straight instead of curving with the fold. For solid or low-frequency
  textures this is barely noticeable; for anything with strong directional pattern
  it will be the next thing someone notices after "why does this look sticker-y" is
  fixed.

## What I'd ship first

Luminance-transfer-only (step 1 + 2, no UV warping), sourced from an approximate
grayscale-of-original-render luminance map if a real shading pass isn't already
available from the rendering pipeline. This is the cheapest version that meaningfully
fixes the "sticker" problem — it requires no new 3D pipeline work, no new model, and
directly targets the complaint stated in the brief (fabric doesn't respect the
render's shading). I would treat true UV-aware pattern warping as the natural
two-days-from-now follow-up, gated on whether the rendering pipeline can actually
export UV/normal data cheaply — that's a real dependency check, not just an
engineering-effort question, so it's worth confirming before committing to it as the
next milestone.
