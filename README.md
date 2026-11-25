# Snap & Shop

AI visual search for Shopify stores that keeps pages fast. A shopper takes a photo of a clothing item and the shop shows similar products. The AI runs in the shopper's browser, so the photo never leaves their device.

Final year project for the B.Sc. (Hons) in Software Development at ATU Galway.

Author: Fionn Qualter (G00428202)
Supervisor: Ihab Salawdeh

## Structure

- `pipeline/`: Python code that prepares the AI model and the product fingerprints (embeddings)
- `engine/`: TypeScript loading engine (planned)
- `demo-shop/`: React demo shop (planned)
- `shopify-app/`: Shopify theme app extension (planned)

## Setup (pipeline)

    cd pipeline
    py -V:3.12 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Data

The project uses the DeepFashion In-shop dataset for testing and training. The dataset is not in this repository because its licence limits it to non-commercial research. To use it, request access from the DeepFashion project page and place the files in `pipeline/data/deepfashion/`.