# SynthOCT Target And Physics Context

## Challenge Target

The challenge asks for a generator that maps a real OCT B-scan to a digital phantom: a point-scatterer distribution with columns `X, Y, Z, Energy`. The organizers run the phantom through a fixed virtual OCT scanner and compare the generated B-scan against the real reference. The official baseline states that leaderboard ranking uses **MS-SSIM** and **LPIPS**, while the challenge description emphasizes physical consistency through OAC and speckle statistics.

## Mathematical Context

- Inverse problem: infer latent scatterer density and backscatter energy from a log-compressed OCT intensity image.
- Non-identifiability: many scatterer fields can produce similar speckle, so the goal is not unique recovery but scanner-consistent synthesis.
- Objective proxy: maximize MS-SSIM and minimize LPIPS after scanning; use OAC/SC/RSC maps as regularizers against visually plausible but physically wrong phantoms.
- Sampling model: scatterer locations are naturally modeled as an inhomogeneous point process with depth- and lateral-dependent intensity.
- Energy model: scanner reflection amplitude scales roughly with `sqrt(Energy/100)`, so energy errors are nonlinear after scanning.
- Cross-validation: improvements must survive grouped folds because optimizing one skin site/frame can overfit texture noise.

## Physics Context

- Beer-Lambert attenuation: OCT signal decays with cumulative optical attenuation, so deeper pixels need compensation when estimating scatterer density.
- Optical attenuation coefficient: OAC estimates local backscattering/attenuation structure and is a stronger physical guide than raw intensity alone.
- Speckle statistics: coherent imaging produces granular texture whose contrast depends on scatterer count, heterogeneity, and resolution-cell occupancy.
- Layer interfaces: skin has axial structure; epidermis/dermis boundaries can produce strong gradients and should influence density/energy allocation.
- Lateral coherence: neighboring A-scans are not independent tissue realizations; excessive lateral noise hurts structural metrics.
- Meso-structures: voids/vessels/inclusions may matter, but naive random voids can hurt if placement is not image-conditioned.

## H11-H40 Rationale

- H11 low depth compensation: tests whether H1 overcompensates deep attenuation; confirmed best so far.
- H12 mid depth compensation: midpoint between H11 and H1.
- H13 high depth compensation: probes whether deeper dermis needs stronger scatterer recovery.
- H14 low OAC weight: tests raw-intensity dominance.
- H15 high OAC weight: tests stronger physical attenuation dominance.
- H16 sublinear density: flattens high-density peaks to reduce over-concentrated scatterers.
- H17 superlinear density: sharpens strong tissue signals; confirmed competitive.
- H18 texture light: reduces speckle contrast contribution.
- H19 texture heavy: increases SC influence for granular matching.
- H20 weak boundary: tests a small layer-interface prior.
- H21 medium boundary: tests stronger interface enhancement.
- H22 low energy scale: avoids over-bright phantom energy.
- H23 high energy scale: tests stronger reflectivity.
- H24 lateral sharp: preserves local lateral details.
- H25 lateral smooth: enforces tissue continuity across A-scans.
- H26 gentle log energy: compresses high OAC-derived energies.
- H27 sparse voids: tests a weak meso-inclusion prior.
- H28 boundary plus voids: tests whether voids are safer near layer transitions.
- H29 low depth plus high OAC: combines H11-style depth with stronger physical OAC; confirmed competitive.
- H30 high depth plus low OAC: opposite interaction test.
- H31 sublinear boundary: soft density plus interface prior.
- H32 superlinear boundary: sharper density plus interface prior; confirmed competitive.
- H33 speckle/OAC balance: joint SC and OAC tuning.
- H34 epidermal emphasis: shallow-layer-biased compensation; confirmed competitive but slower.
- H35 deep dermis emphasis: deeper compensation for highly attenuated scans.
- H36 light multilayer: weak multi-peak axial prior; competitive.
- H37 smooth multilayer: multilayer plus lateral continuity; competitive.
- H38 H1/H5 midpoint: blends attenuation density and boundary band.
- H39 best-pair tuned: hand-tuned local blend around H1/H11/H5.
- H40 conservative winner: low-risk H11-like blend; competitive.

## Current Decision

H11 was the base hypothesis in the invalidated offline ranking. The next mathematically clean move is not to add complexity; it is hosted-API retesting around low or negative depth compensation, OAC weight, and density power, with H61/H67/H68 as current API-facing candidates.
