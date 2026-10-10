// Generates the film's sound effects with ElevenLabs (text-to-sound v2). Already-made files are skipped, so re-runs cost nothing.
//   node --env-file="C:\Users\elija\Downloads\Coding Projects\3D & VR Classrooms\VR\.env" sfx\gen_sfx.mjs
// Writes sfx/<name>.mp3 and sfx/cues.json (prompt + credits used per cue).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const KEY = process.env.ELEVENLABS_API_KEY;
if (!KEY) {
  console.log("ELEVENLABS_API_KEY is not set (run with --env-file pointing at a .env that has it).");
  process.exit(1);
}

const CUES = [
  { name: "ambience", dur: 30, loop: true, infl: 0.4,
    text: "Cold night wind blowing softly around the top of a tall stone clock tower, distant city hum far below, calm and wide, no music" },
  { name: "drone", dur: 14, infl: 0.45,
    text: "Dark mysterious cinematic ambient drone, low warm pad with a faint shimmering high texture, night, slow, no beat" },
  { name: "tick_a", dur: 0.5, infl: 0.75,
    text: "One single sharp tick of a giant antique brass tower clock escapement, crisp metallic click with a wooden thump, close and dry, isolated" },
  { name: "tick_b", dur: 0.5, infl: 0.75,
    text: "One single deep tock of a giant antique tower clock, heavy brass and oak clunk, close and dry, isolated" },
  { name: "tick_c", dur: 0.5, infl: 0.75,
    text: "A single loud mechanical clock tick, big brass escapement wheel locking, sharp transient, isolated, no reverb" },
  { name: "gears", dur: 10, infl: 0.5,
    text: "Inside a giant clock tower mechanism, huge brass gears and cogs slowly turning, rhythmic clicking ratchets, creaking oak beams, low metallic hum" },
  { name: "whoosh", dur: 2, infl: 0.6,
    text: "Deep cinematic whoosh rushing forward through a pane of glass, airy swoosh with a soft shimmering glass texture" },
  { name: "clunk", dur: 2, infl: 0.6,
    text: "Heavy iron clock mechanism clunk as a big lever drops, then a fast ratcheting whirr of gears winding up, tower clock striking train releasing" },
  { name: "bell_a", dur: 10, infl: 0.65,
    text: "One single massive bronze cathedral bell strike at midnight, deep powerful bong with a long rich shimmering reverberating decay" },
  { name: "bell_b", dur: 10, infl: 0.65,
    text: "A huge old church tower bell struck once by a hammer, enormous deep bong, long humming bronze resonance slowly fading" },
  { name: "impact", dur: 4, infl: 0.55,
    text: "Deep cinematic sub bass boom impact with a long low rumbling tail, trailer hit" },
  { name: "birds", dur: 4, infl: 0.55,
    text: "A flock of pigeons suddenly bursting into flight together, many wings flapping loudly, scattering into the night air" },
  { name: "riser", dur: 6, infl: 0.5,
    text: "Dark cinematic tension riser, low strings and sub swell slowly building louder toward a big hit" },
];

async function credits() {
  try {
    const r = await fetch("https://api.elevenlabs.io/v1/user/subscription", { headers: { "xi-api-key": KEY } });
    if (!r.ok) return null;
    const j = await r.json();
    return { used: j.character_count, limit: j.character_limit, tier: j.tier };
  } catch {
    return null;
  }
}

const before = await credits();
if (before) console.log(`account: ${before.tier}, ${before.used} / ${before.limit} credits used`);
const manifestPath = path.join(here, "cues.json");
const manifest = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, "utf8")) : {};

for (const c of CUES) {
  const out = path.join(here, `${c.name}.mp3`);
  if (fs.existsSync(out) && fs.statSync(out).size > 1000) {
    console.log(`skip ${c.name} (exists)`);
    continue;
  }
  const t0 = Date.now();
  const r = await fetch("https://api.elevenlabs.io/v1/sound-generation?output_format=mp3_44100_128", {
    method: "POST",
    headers: { "xi-api-key": KEY, "content-type": "application/json" },
    body: JSON.stringify({ text: c.text, duration_seconds: c.dur, prompt_influence: c.infl, loop: !!c.loop, model_id: "eleven_text_to_sound_v2" }),
  });
  if (!r.ok) {
    console.log(`FAIL ${c.name}: HTTP ${r.status} ${(await r.text()).slice(0, 300)}`);
    continue;
  }
  const buf = Buffer.from(await r.arrayBuffer());
  fs.writeFileSync(out, buf);
  const cost = r.headers.get("character-cost") || r.headers.get("x-character-count") || null;
  manifest[c.name] = { text: c.text, dur: c.dur, loop: !!c.loop, bytes: buf.length, cost };
  console.log(`ok ${c.name}: ${(buf.length / 1024).toFixed(0)} KB in ${((Date.now() - t0) / 1000).toFixed(1)} s${cost ? `, cost ${cost}` : ""}`);
}
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
const after = await credits();
if (before && after) console.log(`credits spent this run: ${after.used - before.used} (now ${after.used} / ${after.limit})`);
