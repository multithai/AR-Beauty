/**
 * Cloudflare Worker — AI beauty retouch proxy for the DECODE YOUR SKIN booth.
 *
 * It holds the Replicate API token (never exposed to the browser), receives a
 * captured photo, runs it through a generative face-retouch model, and returns
 * the beautified image URL.
 *
 * Secrets / vars (set with `wrangler secret put` or in the CF dashboard):
 *   REPLICATE_API_TOKEN   (required)  your Replicate token (r8_...)
 *   REPLICATE_MODEL       (optional)  "owner/name", default flux-kontext-pro
 *   BEAUTY_PROMPT         (optional)  edit instruction for the model
 *   ALLOW_ORIGIN          (optional)  allowed web origin, default "*"
 *
 * Endpoint: POST { image: "data:image/png;base64,..." }  ->  { image: "<url>" }
 */

const DEFAULT_MODEL = "black-forest-labs/flux-kontext-pro";
const DEFAULT_PROMPT =
  "Retouch this portrait like a professional beauty photo: smooth and even " +
  "the skin, remove blemishes and dark spots, add a healthy natural glow, " +
  "brighten and clarify the complexion. Keep the same person, identity, face " +
  "shape, features and pose exactly the same. Photorealistic, natural skin texture.";

function cors(origin) {
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

export default {
  async fetch(request, env) {
    const origin = env.ALLOW_ORIGIN || "*";
    const headers = { ...cors(origin), "Content-Type": "application/json" };

    if (request.method === "OPTIONS") return new Response(null, { headers: cors(origin) });
    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "POST only" }), { status: 405, headers });
    }
    if (!env.REPLICATE_API_TOKEN) {
      return new Response(JSON.stringify({ error: "Server missing REPLICATE_API_TOKEN" }), { status: 500, headers });
    }

    let body;
    try { body = await request.json(); } catch { body = null; }
    const image = body && body.image;
    if (!image) {
      return new Response(JSON.stringify({ error: "Missing 'image'" }), { status: 400, headers });
    }

    const model = env.REPLICATE_MODEL || DEFAULT_MODEL;
    const prompt = env.BEAUTY_PROMPT || DEFAULT_PROMPT;

    // Input shape for FLUX-Kontext style edit models. If you switch to a
    // restoration model like tencentarc/gfpgan, change this to { img: image }.
    const input = {
      prompt,
      input_image: image,
      output_format: "jpg",
      safety_tolerance: 2,
    };

    try {
      // `Prefer: wait` makes Replicate hold the request open until the
      // prediction finishes (up to 60s) so we don't have to poll.
      const res = await fetch(`https://api.replicate.com/v1/models/${model}/predictions`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.REPLICATE_API_TOKEN}`,
          "Content-Type": "application/json",
          Prefer: "wait",
        },
        body: JSON.stringify({ input }),
      });

      const data = await res.json();
      if (!res.ok) {
        return new Response(JSON.stringify({ error: data.detail || "Replicate error", raw: data }), { status: 502, headers });
      }

      // output can be a string URL or an array of URLs
      let out = data.output;
      if (Array.isArray(out)) out = out[out.length - 1];
      if (!out) {
        return new Response(JSON.stringify({ error: "No output", status: data.status, raw: data }), { status: 502, headers });
      }

      return new Response(JSON.stringify({ image: out }), { headers });
    } catch (e) {
      return new Response(JSON.stringify({ error: String(e) }), { status: 500, headers });
    }
  },
};
