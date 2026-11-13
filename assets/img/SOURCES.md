# Image sources

## Model-developer icons (`logos/`)

LobeHub's lobe-icons, npm package `@lobehub/icons-static-svg` 1.101.0 (https://github.com/lobehub/lobe-icons),
the same files as `paper/logos/lobehub/`. `anthropic.svg`, `openai.svg` and `zai.svg` are the monochrome icons;
`deepseek.svg`, `kimi.svg` and `minimax.svg` are the `-color` variants. MIT license, "Copyright (c) 2023 LobeHub"
(full text in `logos/LICENSE.txt`). The logos themselves are trademarks of their owners.

## Sponsor logos (`sponsors/`)

- `nvidia.png`: `paper/assets/nvidia-logo.png` (the logo in the paper's acknowledgments), cropped to the logo and
  with its white background made transparent.
- `fireworks.svg`: the icon and wordmark from the header of https://fireworks.ai/, placed side by side as on that
  page, in the color of the paper's Fireworks icon (`#280016`).
- `daytona.svg`: the black wordmark from Daytona's brand page, https://www.daytona.io/brand
  (https://framerusercontent.com/images/Sg9zE3EMDPr8DFon0ERhWoHkso.svg), unchanged.
- `snorkel.svg`: the light-theme logo from https://snorkel.ai/
  (https://s46486.pcdn.co/wp-content/uploads/2022/01/snorkel-light-logo.svg), unchanged.

## Affiliation logos (`orgs/`)

The logos at the top of the paper's first page (`paper/assets/logos/*.pdf`), converted to SVG with
`pdftocairo -svg`. Their sources, as recorded in `paper/assets/logos/LOGO_SOURCES.txt`:

- `infini-ai-lab.svg`: the Infini AI Lab infinity mark with "Infini AI Lab" set in Hanken Grotesk 600, made for
  the paper (`paper/assets/logos/infini-ai-lab.pdf`). It sits above the CMU logo, as in the paper.
- `cmu.svg`, `nvidia.svg`, `amazon.svg` (for Amazon AGI), `uchicago.svg`, `meta.svg`, `ucsb.svg` and `uiuc.svg`:
  Wikimedia Commons SVGs.
- `radixark.svg` and `ut-austin.svg`: the image files in the source of arXiv:2611.11995.
- `hkust.svg`: https://hkust.edu.hk/sites/default/files/2024-03/HKUST_logo_1.svg
- `together.svg`: the header SVG of https://www.together.ai/
- `alation.svg`: the header SVG of https://www.alation.com/

The logos are trademarks of their owners and appear here, as in the paper, to identify the authors' affiliations.

## Paper figures (`figures/`)

Downscaled copies of the paper's figures: `serve_latency_throughput.png`, `pretrain_params_tokens.png`,
`midtrain_mixture.png` and `cluster_attainment_completion.png` come from `paper/figures/`;
`task_execution_architecture.png` is `paper/figures/task_execution_architecture.drawio.pdf` rendered at 150 dpi.

## Social preview (`card.png`)

A 1200 × 630 screenshot of this site's hero, used for `og:image` and `twitter:image`.
