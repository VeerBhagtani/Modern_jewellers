/* Arouse in the browser.
 *
 * A JavaScript port of the Arouse runtime: byte-level BPE tokenizer, the decoder-only
 * Transformer (RoPE, grouped-query attention, SwiGLU, RMSNorm, KV cache), sampling,
 * the action protocol codec, copy/date-constrained decoding, grounding + answer guards,
 * the agent loop, and the sandbox tools. It mirrors the Python implementation in
 * arouse/ (parity is tested in tests/test_web.py). No network calls except loading the
 * model files that ship with the page.
 */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- constants
  const S = { PAD: 0, BOS: 1, EOS: 2, END: 3, SYSTEM: 4, TOOLS: 5, CONTEXT: 6, MEMORY: 7, STATE: 8, USER: 9,
    TOOL_RESULT: 10, TOOL_ERROR: 11, AROUSE: 12, PLAN: 13, VERIFY: 14, TOOL_CALL: 15, ASK_USER: 16, FINISH: 17, FAIL: 18 };
  const NUM_SPECIAL = 64, BYTE_OFFSET = 64, FIRST_MERGE = 320;
  const ACTION_TOKEN = { tool_call: S.TOOL_CALL, ask_user: S.ASK_USER, finish: S.FINISH, fail: S.FAIL };
  const TOKEN_ACTION = { 15: "tool_call", 16: "ask_user", 17: "finish", 18: "fail" };
  const INPUT_SEGMENTS = [S.SYSTEM, S.TOOLS, S.CONTEXT, S.MEMORY, S.STATE, S.USER, S.TOOL_RESULT, S.TOOL_ERROR];
  const REQUIRED = { tool_call: ["tool", "arguments"], ask_user: ["question"], finish: ["result"], fail: ["error"] };
  const SYSTEM_PROMPT = "You are Arouse. Complete tasks with tools. Ask when a detail is missing. " +
    "Never claim success without a successful tool result.";
  const WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
  const DAY_CODES = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"];
  const MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december"];
  const enc = new TextEncoder();
  const decLoose = new TextDecoder("utf-8");
  const decStrict = new TextDecoder("utf-8", { fatal: true });

  class ProtocolError extends Error {}

  // ---------------------------------------------------------------- tokenizer
  // Same chunking as arouse/tokenizer/pretokenize.py (PATTERN_V1), written with Unicode classes.
  const PRETOKENIZE = new RegExp(
    "[^\\r\\n\\p{L}\\p{N}_\\p{M}]?[\\p{L}\\p{Nl}\\p{No}\\p{M}]+" +
    "|_+[\\p{L}\\p{Nl}\\p{No}\\p{M}]*" +
    "|\\p{Nd}" +
    "| ?[^\\s\\p{L}\\p{N}_\\p{M}]+[\\r\\n]*" +
    "|\\s*[\\r\\n]+" +
    "|\\s+(?!\\S)" +
    "|\\s+" +
    "|[\\s\\S]", "gu");

  function concatBytes(a, b) {
    const out = new Uint8Array(a.length + b.length);
    out.set(a, 0); out.set(b, a.length);
    return out;
  }

  class Tokenizer {
    constructor(spec) {
      this.special = spec.special_tokens;
      this.specialToId = new Map(this.special.map((t, i) => [t, i]));
      this.bytes = this.special.map((t) => enc.encode(t));
      for (let b = 0; b < 256; b++) this.bytes.push(Uint8Array.of(b));
      this.ranks = new Map();
      spec.merges.forEach(([a, b], r) => {
        this.ranks.set(a * 8192 + b, r);
        this.bytes.push(concatBytes(this.bytes[a], this.bytes[b]));
      });
      this.vocabSize = this.bytes.length;
      this.cache = new Map();
    }
    isSpecial(id) { return id >= 0 && id < NUM_SPECIAL; }
    bpe(chunk) {
      let ids = Array.from(enc.encode(chunk), (b) => b + BYTE_OFFSET);
      while (ids.length >= 2) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < ids.length - 1; i++) {
          const r = this.ranks.get(ids[i] * 8192 + ids[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        const a = ids[best], b = ids[best + 1], merged = FIRST_MERGE + bestRank, out = [];
        for (let i = 0; i < ids.length;) {
          if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(merged); i += 2; }
          else { out.push(ids[i]); i += 1; }
        }
        ids = out;
      }
      return ids;
    }
    /** Plain text -> ids. Never produces special tokens (injection-safe). */
    encode(text) {
      const out = [];
      for (const m of text.matchAll(PRETOKENIZE)) {
        let ids = this.cache.get(m[0]);
        if (!ids) {
          ids = this.bpe(m[0]);
          if (this.cache.size > 50000) this.cache.clear();
          this.cache.set(m[0], ids);
        }
        for (const id of ids) out.push(id);
      }
      return out;
    }
    idBytes(id) { return this.bytes[id]; }
    decode(ids) {
      let n = 0;
      for (const id of ids) n += this.bytes[id].length;
      const buf = new Uint8Array(n);
      let o = 0;
      for (const id of ids) { buf.set(this.bytes[id], o); o += this.bytes[id].length; }
      return decLoose.decode(buf);
    }
  }

  // ---------------------------------------------------------------- model
  function halfToFloatTable() {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      t[h] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
        : e === 31 ? (f ? NaN : s * Infinity)
          : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return t;
  }

  const BLOCK = 32; // prompt tokens processed together (weights are read once per block)

  // Y[t, r] = sum_c W[r, c] * X[t, c] for t < T (W is rows x cols, row-major)
  function matmul(W, X, Y, T, rows, cols) {
    for (let r = 0; r < rows; r++) {
      const base = r * cols;
      let t = 0;
      for (; t + 3 < T; t += 4) { // four tokens share each weight load
        const x0 = t * cols, x1 = x0 + cols, x2 = x1 + cols, x3 = x2 + cols;
        let a = 0, b = 0, c2 = 0, e = 0;
        for (let c = 0; c < cols; c++) {
          const w = W[base + c];
          a += w * X[x0 + c]; b += w * X[x1 + c]; c2 += w * X[x2 + c]; e += w * X[x3 + c];
        }
        Y[t * rows + r] = a; Y[(t + 1) * rows + r] = b; Y[(t + 2) * rows + r] = c2; Y[(t + 3) * rows + r] = e;
      }
      for (; t < T; t++) {
        const xo = t * cols;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0, c = 0;
        for (; c + 3 < cols; c += 4) {
          s0 += W[base + c] * X[xo + c]; s1 += W[base + c + 1] * X[xo + c + 1];
          s2 += W[base + c + 2] * X[xo + c + 2]; s3 += W[base + c + 3] * X[xo + c + 3];
        }
        for (; c < cols; c++) s0 += W[base + c] * X[xo + c];
        Y[t * rows + r] = s0 + s1 + s2 + s3;
      }
    }
  }

  function rmsnormRows(X, w, Y, T, d, eps) {
    for (let t = 0; t < T; t++) {
      const o = t * d;
      let ss = 0;
      for (let i = 0; i < d; i++) ss += X[o + i] * X[o + i];
      const inv = 1 / Math.sqrt(ss / d + eps);
      for (let i = 0; i < d; i++) Y[o + i] = X[o + i] * inv * w[i];
    }
  }

  class Model {
    constructor(cfg, buffer) {
      this.cfg = cfg;
      const table = halfToFloatTable(), half = new Uint16Array(buffer);
      const get = (name) => {
        const t = cfg.tensors.find((x) => x.name === name);
        const n = t.shape.reduce((a, b) => a * b, 1), out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = table[half[t.offset + i]];
        return out;
      };
      this.d = cfg.d_model; this.H = cfg.n_heads; this.KV = cfg.n_kv_heads; this.hd = this.d / this.H;
      this.f = cfg.ffn_hidden_size; this.V = cfg.vocab_size; this.ctx = cfg.context_length; this.eps = cfg.norm_eps;
      this.emb = get("embed.weight");
      this.head = cfg.tie_embeddings ? this.emb : get("lm_head.weight");
      this.layers = [];
      for (let i = 0; i < cfg.n_layers; i++) {
        const p = `layers.${i}.`;
        this.layers.push({
          an: get(p + "attn_norm.weight"), wq: get(p + "attn.wq.weight"), wk: get(p + "attn.wk.weight"),
          wv: get(p + "attn.wv.weight"), wo: get(p + "attn.wo.weight"), mn: get(p + "mlp_norm.weight"),
          wg: get(p + "mlp.w_gate.weight"), wu: get(p + "mlp.w_up.weight"), wd: get(p + "mlp.w_down.weight"),
          kc: new Float32Array(this.ctx * this.KV * this.hd), vc: new Float32Array(this.ctx * this.KV * this.hd),
        });
      }
      this.norm = get("norm.weight");
      // RoPE tables (half-split rotation, like arouse/model/rope.py)
      const half2 = this.hd / 2;
      this.cos = new Float32Array(this.ctx * half2); this.sin = new Float32Array(this.ctx * half2);
      for (let pos = 0; pos < this.ctx; pos++) {
        for (let i = 0; i < half2; i++) {
          const inv = 1 / Math.pow(cfg.rope_theta, (2 * i) / this.hd), ang = pos * inv;
          this.cos[pos * half2 + i] = Math.cos(ang); this.sin[pos * half2 + i] = Math.sin(ang);
        }
      }
      const B = BLOCK, d = this.d, hd = this.hd;
      this.x = new Float32Array(B * d); this.h = new Float32Array(B * d); this.o = new Float32Array(B * d);
      this.q = new Float32Array(B * this.H * hd); this.k = new Float32Array(B * this.KV * hd); this.v = new Float32Array(B * this.KV * hd);
      this.att = new Float32Array(B * this.H * hd); this.scores = new Float32Array(this.ctx);
      this.g = new Float32Array(B * this.f); this.u = new Float32Array(B * this.f);
      this.logits = new Float32Array(this.V);
      this.len = 0;
    }
    rope(vec, off, heads, pos) {
      const hd = this.hd, half = hd / 2, c = this.cos, s = this.sin, base = pos * half;
      for (let h = 0; h < heads; h++) {
        const o = off + h * hd;
        for (let i = 0; i < half; i++) {
          const x1 = vec[o + i], x2 = vec[o + i + half], cs = c[base + i], sn = s[base + i];
          vec[o + i] = x1 * cs - x2 * sn;
          vec[o + i + half] = x1 * sn + x2 * cs;
        }
      }
    }
    /** Feed one token at position this.len; returns logits for the next token. */
    step(tok) { return this.forward([tok]); }
    /** Feed tokens at positions this.len...; returns logits after the last one. */
    forward(tokens) {
      for (let s = 0; s < tokens.length; s += BLOCK) this.block(tokens.slice(s, s + BLOCK), s + BLOCK >= tokens.length);
      return this.logits;
    }
    block(toks, last) {
      const T = toks.length, { d, H, KV, hd, f } = this, p0 = this.len, rep = H / KV, scale = 1 / Math.sqrt(hd), kvw = KV * hd;
      if (p0 + T > this.ctx) throw new ProtocolError("context full");
      const { x, h, o, q, k, v, att, g, u, scores } = this, qw = H * hd;
      for (let t = 0; t < T; t++) x.set(this.emb.subarray(toks[t] * d, toks[t] * d + d), t * d);
      for (const L of this.layers) {
        rmsnormRows(x, L.an, h, T, d, this.eps);
        matmul(L.wq, h, q, T, qw, d); matmul(L.wk, h, k, T, kvw, d); matmul(L.wv, h, v, T, kvw, d);
        for (let t = 0; t < T; t++) {
          const pos = p0 + t;
          this.rope(q, t * qw, H, pos); this.rope(k, t * kvw, KV, pos);
          L.kc.set(k.subarray(t * kvw, (t + 1) * kvw), pos * kvw); L.vc.set(v.subarray(t * kvw, (t + 1) * kvw), pos * kvw);
        }
        for (let t = 0; t < T; t++) {
          const pos = p0 + t;
          for (let qh = 0; qh < H; qh++) {
            const kh = (qh / rep) | 0, qo = t * qw + qh * hd;
            let mx = -Infinity;
            for (let p = 0; p <= pos; p++) {
              const ko = p * kvw + kh * hd;
              let sc = 0;
              for (let i = 0; i < hd; i++) sc += q[qo + i] * L.kc[ko + i];
              sc *= scale; scores[p] = sc; if (sc > mx) mx = sc;
            }
            let sum = 0;
            for (let p = 0; p <= pos; p++) { const e = Math.exp(scores[p] - mx); scores[p] = e; sum += e; }
            for (let i = 0; i < hd; i++) att[qo + i] = 0;
            for (let p = 0; p <= pos; p++) {
              const w = scores[p] / sum, vo = p * kvw + kh * hd;
              for (let i = 0; i < hd; i++) att[qo + i] += w * L.vc[vo + i];
            }
          }
        }
        matmul(L.wo, att, o, T, d, qw);
        for (let i = 0; i < T * d; i++) x[i] += o[i];
        rmsnormRows(x, L.mn, h, T, d, this.eps);
        matmul(L.wg, h, g, T, f, d); matmul(L.wu, h, u, T, f, d);
        for (let i = 0; i < T * f; i++) { const gv = g[i]; g[i] = (gv / (1 + Math.exp(-gv))) * u[i]; }
        matmul(L.wd, g, o, T, d, f);
        for (let i = 0; i < T * d; i++) x[i] += o[i];
      }
      this.len += T;
      if (last) { // logits only for the final position
        rmsnormRows(x.subarray((T - 1) * d, T * d), this.norm, h, 1, d, this.eps);
        matmul(this.head, h, this.logits, 1, this.V, d);
      }
    }
  }

  // ---------------------------------------------------------------- engine
  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleNext(logits, temperature, topP, rand) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    if (temperature === 0) return best;
    const idx = [], vals = [];
    for (let i = 0; i < logits.length; i++) if (Number.isFinite(logits[i])) { idx.push(i); vals.push(logits[i] / temperature); }
    const mx = Math.max(...vals), probs = vals.map((v) => Math.exp(v - mx)), sum = probs.reduce((a, b) => a + b, 0);
    const order = probs.map((p, i) => [p / sum, idx[i]]).sort((a, b) => b[0] - a[0]);
    let cum = 0;
    const keep = [];
    for (const [p, id] of order) { if (cum >= topP) break; keep.push([p, id]); cum += p; }
    const tot = keep.reduce((a, b) => a + b[0], 0);
    let r = rand() * tot;
    for (const [p, id] of keep) { r -= p; if (r <= 0) return id; }
    return keep[keep.length - 1][1];
  }

  class Engine {
    constructor(model, tok) {
      this.model = model; this.tok = tok; this.cached = [];
      const named = new Set(Object.values(S));
      this.banned = [S.PAD, S.BOS, S.AROUSE, ...INPUT_SEGMENTS];
      for (let i = 0; i < NUM_SPECIAL; i++) if (!named.has(i)) this.banned.push(i);
      for (let i = tok.vocabSize; i < model.V; i++) this.banned.push(i);
    }
    get contextLength() { return this.model.ctx; }
    /** Generate after `prompt`; reuses the KV cache for any shared prefix with the previous call. */
    generate(prompt, { maxNew = 200, temperature = 0, topP = 1, seed = 0, stop = [S.END], banned = [], hook = null } = {}) {
      const m = this.model;
      if (prompt.length >= m.ctx) return { ids: [], text: "", finish: "context" };
      let common = 0;
      while (common < this.cached.length && common < prompt.length - 1 && this.cached[common] === prompt[common]) common++;
      m.len = common; this.cached.length = common;
      let logits = m.forward(prompt.slice(common));
      this.cached.push(...prompt.slice(common));
      const rand = mulberry32(seed + 1), stopSet = new Set(stop), out = [];
      const ban = this.banned.concat(banned);
      let finish = "length";
      for (let n = 0; n < maxNew; n++) {
        const lg = Float32Array.from(logits);
        for (const b of ban) lg[b] = -Infinity;
        const masked = hook ? hook(out, lg) : lg;
        const t = sampleNext(masked, temperature, topP, rand);
        if (stopSet.has(t)) { finish = "stop"; break; }
        out.push(t);
        if (m.len + 1 >= m.ctx) { finish = "context"; break; }
        if (n === maxNew - 1) break;
        logits = m.step(t); this.cached.push(t);
      }
      return { ids: out, text: this.tok.decode(out), finish };
    }
  }

  // ---------------------------------------------------------------- protocol
  function sortedValue(v) {
    if (Array.isArray(v)) return v.map(sortedValue);
    if (v && typeof v === "object") {
      const o = {};
      for (const k of Object.keys(v).sort()) o[k] = sortedValue(v[k]);
      return o;
    }
    return v;
  }
  /** Same bytes as arouse.protocol.canonical_json: sorted keys, "tool" first, compact. */
  function canonicalJson(obj) {
    let v = sortedValue(obj);
    if (v && typeof v === "object" && !Array.isArray(v) && "tool" in v) {
      const o = { tool: v.tool };
      for (const k of Object.keys(v)) if (k !== "tool") o[k] = v[k];
      v = o;
    }
    return JSON.stringify(v);
  }

  const TOOL_NAME = /^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/;
  function validateAction(d) {
    if (!d || typeof d !== "object" || Array.isArray(d)) throw new ProtocolError("action must be an object");
    const req = REQUIRED[d.type];
    if (!req) throw new ProtocolError(`bad action type ${d.type}`);
    for (const k of Object.keys(d)) if (k !== "type" && !req.includes(k)) throw new ProtocolError(`unexpected field ${k}`);
    for (const k of req) if (d[k] === undefined || d[k] === null) throw new ProtocolError(`missing ${k}`);
    if (d.type === "tool_call") {
      if (typeof d.tool !== "string" || !TOOL_NAME.test(d.tool)) throw new ProtocolError("bad tool name");
      if (!d.arguments || typeof d.arguments !== "object" || Array.isArray(d.arguments)) throw new ProtocolError("arguments must be an object");
    } else if (typeof d[req[0]] !== "string" || !d[req[0]].trim()) throw new ProtocolError(`${req[0]} must be non-empty`);
    return d;
  }
  function actionBody(a) { const o = {}; for (const k of REQUIRED[a.type]) o[k] = a[k]; return o; }

  // Tool argument schemas (arouse/agent/tools.py)
  const P = (type, extra = {}) => ({ type, required: true, ...extra });
  const DATE_RE = /^\d{4}-\d{2}-\d{2}$/, TIME_RE = /^([01]\d|2[0-3]):[0-5]\d$/;
  const REPEAT = P("object", { required: false, properties: {
    freq: P("string", { enum: ["daily", "weekly", "monthly"] }),
    interval: P("integer", { required: false, minimum: 1, maximum: 365 }),
    by_day: P("array", { required: false, items: P("string", { enum: DAY_CODES }) }),
    by_month_day: P("array", { required: false, items: P("integer", { minimum: 1, maximum: 28 }) }),
  } });
  const TOOLS = {
    "scheduler.create": { task: P("string"), date: P("string", { required: false, pattern: DATE_RE }),
      time: P("string", { required: false, pattern: TIME_RE }),
      in_minutes: P("integer", { required: false, minimum: 1, maximum: 10080 }), repeat: REPEAT },
    "scheduler.list": {}, "scheduler.delete": { task_id: P("string") }, "notes.create": { text: P("string") },
    "file.read": { path: P("string") }, "file.list": {},
    "kb.search": { query: P("string") },
    "gst.calculate": { amount: P("string"), rate: P("string"), inclusive: P("boolean", { required: false }) },
    "leads.find": { method: P("string", { enum: ["inactive", "occasions", "top", "custom"] }), query: P("string", { required: false }) },
  };
  const TOOL_NAMES = Object.keys(TOOLS);
  function checkParam(p, v, path) {
    const ok = { string: typeof v === "string", integer: Number.isInteger(v), object: v && typeof v === "object" && !Array.isArray(v),
      array: Array.isArray(v), boolean: typeof v === "boolean" }[p.type];
    if (!ok) throw new ProtocolError(`${path}: expected ${p.type}`);
    if (p.enum && !p.enum.includes(v)) throw new ProtocolError(`${path}: not allowed`);
    if (p.type === "string") {
      if (!v.trim()) throw new ProtocolError(`${path}: must not be empty`);
      if (p.pattern && !p.pattern.test(v)) throw new ProtocolError(`${path}: invalid format`);
    }
    if (p.type === "integer") {
      if (p.minimum !== undefined && v < p.minimum) throw new ProtocolError(`${path}: too small`);
      if (p.maximum !== undefined && v > p.maximum) throw new ProtocolError(`${path}: too large`);
    }
    if (p.type === "array" && p.items) { if (!v.length) throw new ProtocolError(`${path}: empty`); v.forEach((x, i) => checkParam(p.items, x, `${path}[${i}]`)); }
    if (p.type === "object" && p.properties) checkObject(p.properties, v, path);
  }
  function checkObject(props, v, path) {
    for (const k of Object.keys(v)) if (!(k in props)) throw new ProtocolError(`${path}: unexpected ${k}`);
    for (const [k, p] of Object.entries(props)) {
      if (k in v) checkParam(p, v[k], `${path}.${k}`);
      else if (p.required) throw new ProtocolError(`${path}: missing ${k}`);
    }
  }
  function validateCall(tool, args) {
    if (!(tool in TOOLS)) throw new ProtocolError(`unknown tool ${tool}`);
    checkObject(TOOLS[tool], args, tool);
  }

  function encodeTurnBody(tok, turn) {
    const ids = [];
    if (turn.plan) ids.push(S.PLAN, ...tok.encode(turn.plan));
    if (turn.verify) ids.push(S.VERIFY, ...tok.encode(turn.verify));
    ids.push(ACTION_TOKEN[turn.action.type], ...tok.encode(canonicalJson(actionBody(turn.action))), S.END);
    return ids;
  }

  function decodeTurn(tok, ids) {
    if (ids.length && ids[ids.length - 1] === S.END) ids = ids.slice(0, -1);
    const sections = [];
    for (const t of ids) {
      if (tok.isSpecial(t)) sections.push([t, []]);
      else if (!sections.length) throw new ProtocolError("turn must start with a marker");
      else sections[sections.length - 1][1].push(t);
    }
    let plan = null, verify = null, action = null;
    sections.forEach(([marker, body], i) => {
      const text = tok.decode(body);
      if (marker === S.PLAN && plan === null && !action) plan = text.trim();
      else if (marker === S.VERIFY && verify === null && !action) verify = text.trim();
      else if (TOKEN_ACTION[marker] && !action && i === sections.length - 1) {
        let fields;
        try { fields = JSON.parse(text); } catch (e) { throw new ProtocolError("action body is not valid JSON"); }
        if (!fields || typeof fields !== "object" || Array.isArray(fields) || "type" in fields) throw new ProtocolError("bad action body");
        action = validateAction({ type: TOKEN_ACTION[marker], ...fields });
      } else throw new ProtocolError("unexpected token in turn");
    });
    if (!action) throw new ProtocolError("turn has no action");
    return { action, plan: plan || null, verify: verify || null };
  }

  // ---------------------------------------------------------------- context + episodes
  const pad = (n) => String(n).padStart(2, "0");
  const fmtDate = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const fmtDT = (d) => `${fmtDate(d)}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const pyWeekday = (d) => (d.getDay() + 6) % 7; // Monday = 0, like Python
  const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x; };
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function calendarText(now) {
    const parts = [`today ${fmtDate(now)} (${cap(WEEKDAYS[pyWeekday(now)])})`, `tomorrow ${fmtDate(addDays(now, 1))}`];
    for (let k = 1; k <= 7; k++) { const d = addDays(now, k); parts.push(`${cap(WEEKDAYS[pyWeekday(d)])} ${fmtDate(d)}`); }
    return parts.join(", ");
  }
  function buildContext(now, timezone) { return { now: fmtDT(now), timezone, calendar: calendarText(now) }; }

  function segment(tok, marker, text) { return [marker, ...tok.encode(text), S.END]; }
  function encodePrompt(tok, header, events) {
    const ids = [S.BOS, ...segment(tok, S.SYSTEM, header.system || SYSTEM_PROMPT),
      ...segment(tok, S.CONTEXT, canonicalJson(header.context)), ...segment(tok, S.TOOLS, canonicalJson(header.tools))];
    if (header.memory) ids.push(...segment(tok, S.MEMORY, header.memory));
    if (header.state) ids.push(...segment(tok, S.STATE, canonicalJson(header.state)));
    for (const ev of events) {
      if (ev.type === "user") ids.push(...segment(tok, S.USER, ev.content));
      else if (ev.type === "arouse") ids.push(S.AROUSE, ...encodeTurnBody(tok, ev.turn));
      else ids.push(...segment(tok, ev.type === "tool_result" ? S.TOOL_RESULT : S.TOOL_ERROR, canonicalJson(ev.content)));
    }
    ids.push(S.AROUSE);
    return ids;
  }

  // ---------------------------------------------------------------- what the user wrote (arouse/agent/mentions.py)
  const MONEY_RE = /(?:₹\s?|rs\.?\s?|inr\s?)?[0-9][0-9,]*(?:\.[0-9]+)?(?:\s?(?:k|thousand|lakhs?|lacs?|crores?|cr)\b)?(?:\s?rupees)?/gi;
  const PERCENT_AFTER = /\s?(?:%|percent\b|per cent\b)/iy;
  const RATE_IN = /([0-9]+(?:\.[0-9]+)?)\s?(?:%|percent\b|per cent\b)/gi;
  const NUMBER_IN = /[0-9]+(?:\.[0-9]+)?/g;
  function moneyAndRates(said, rateReplies) {
    const amounts = new Set(), rates = new Set();
    for (const text of said) {
      const reply = rateReplies.includes(text);
      for (const m of text.matchAll(MONEY_RE)) {
        const span = m[0].replace(/,+$/, "").trim();
        PERCENT_AFTER.lastIndex = m.index + m[0].length;
        if (!reply && !PERCENT_AFTER.test(text) && span) amounts.add(span);
      }
      for (const m of text.matchAll(RATE_IN)) rates.add(m[1]);
      if (reply) for (const m of text.matchAll(NUMBER_IN)) rates.add(m[0]);
    }
    return [amounts, rates];
  }
  const TIME_PATTERNS = [
    [/\b(\d{1,2})(?::(\d{2}))?\s?(a\.?m\.?|p\.?m\.?)(?![a-z])/gi, "ampm"],
    [/\b(\d{1,2})(?::(\d{2}))?\s+in the (morning|afternoon|evening)\b/gi, "period"],
    [/\b(\d{1,2})(?::(\d{2}))?\s+at night\b/gi, "night"],
    [/\b(\d{1,2}):(\d{2})\b(?!\s?(?:a\.?m|p\.?m))/gi, "clock"],
    [/\bat (\d{1,2})\b(?![:.,]?\d|\s?(?:a\.?m|p\.?m|%)|\s+(?:in the|at night|minutes?|hours?|days?))/gi, "bare"],
  ];
  function mentionedTimes(texts) {
    const out = new Set();
    for (const text of texts) {
      if (/\bnoon\b/i.test(text)) out.add("12:00");
      if (/\bmidnight\b/i.test(text)) out.add("00:00");
      const taken = [];
      for (const [pat, kind] of TIME_PATTERNS) {
        for (const m of text.matchAll(pat)) {
          const a0 = m.index, b0 = m.index + m[0].length;
          if (taken.some(([a, b]) => a0 < b && a < b0)) continue;
          taken.push([a0, b0]);
          let h = Number(m[1]);
          const mm = kind !== "bare" ? Number(m[2] || 0) : 0;
          if (mm > 59 || h > 23) continue;
          let cands;
          if (kind === "ampm") {
            if (h === 0 || h > 12) continue;
            cands = [(h % 12) + (m[3].toLowerCase().startsWith("p") ? 12 : 0)];
          } else if (kind === "period") cands = h >= 1 && h <= 12 ? [(h % 12) + (m[3].toLowerCase() === "morning" ? 0 : 12)] : [];
          else if (kind === "night") cands = h >= 7 && h <= 11 ? [(h % 12) + 12] : [];
          else if (kind === "clock" && (h >= 13 || m[1].startsWith("0") || h === 0)) cands = [h];
          else cands = h >= 1 && h <= 11 ? [h, h + 12] : h <= 23 ? [h] : [];
          for (const c of cands) if (c <= 23) out.add(`${pad(c)}:${pad(mm)}`);
        }
      }
    }
    return out;
  }
  const Q_GST_RATE = "Which GST rate should I use? For example 3% for gold, 5% or 18%.";
  function gstMentions(request) {
    const said = request.filter((e) => e.type === "user").map((e) => e.content), replies = [];
    for (let i = 0; i + 1 < request.length; i++) {
      const ev = request[i], nxt = request[i + 1];
      if (ev.type === "arouse" && ev.turn.action.question === Q_GST_RATE && nxt.type === "user") replies.push(nxt.content);
    }
    return moneyAndRates(said, replies);
  }

  // ---------------------------------------------------------------- grounding
  const WORD_RE = /[\p{L}\p{N}_'’-]+|[^\p{L}\p{N}_\s]/gu;
  const BOUNDARY = new Set(["at", "on", "in", "by", "after", "before", "from", "tomorrow", "today", "tonight", "every", "each",
    "this", "next", "daily", "please", "pls", "thanks", "thank", "thx", "remind", "so", "then", "monthly", "weekly",
    ...WEEKDAYS, ...MONTHS, ...MONTHS.map((m) => m.slice(0, 3))]);
  const COPY_FIELDS = { "scheduler.create": "task", "notes.create": "text", "kb.search": "query", "leads.find": "query" };
  const words = (s) => s.toLowerCase().match(WORD_RE) || [];

  function currentRequest(events) {
    let start = 0;
    events.forEach((ev, i) => { if (ev.type === "arouse" && ["finish", "fail"].includes(ev.turn.action.type)) start = i + 1; });
    return events.slice(start);
  }
  function copiedFrom(value, messages) {
    const v = words(value);
    if (!v.length) return false;
    for (const msg of messages) {
      const w = words(msg);
      for (let i = 0; i + v.length <= w.length; i++) {
        let ok = true;
        for (let j = 0; j < v.length; j++) if (w[i + j] !== v[j]) { ok = false; break; }
        if (!ok) continue;
        const nxt = w[i + v.length];
        if (nxt === undefined || !/^[\p{L}\p{N}]/u.test(nxt) || BOUNDARY.has(nxt)) return true;
      }
    }
    return false;
  }
  const STOPWORDS = new Set(["the", "a", "an", "to", "my", "for", "of", "on", "in", "at", "and", "me", "about", "reminder",
    "reminders", "it", "that", "this", "please", "i", "you"]);
  const content = (text) => new Set(words(text).filter((w) => /^[\p{L}\p{N}]/u.test(w) && !STOPWORDS.has(w)));
  function deleteIssue(taskId, events) {
    const req = currentRequest(events), listed = new Map();
    for (const ev of req) {
      if (ev.type !== "tool_result") continue;
      for (const r of ev.content.reminders || []) listed.set(r.task_id, r.task);
      if ("task_id" in ev.content && "task" in ev.content) listed.set(ev.content.task_id, ev.content.task);
    }
    if (!listed.has(taskId)) return `task_id ${taskId} was not listed in this request`;
    for (const scope of [req, events]) {
      const said = new Set();
      for (const ev of scope) if (ev.type === "user") for (const w of content(ev.content)) said.add(w);
      const score = new Map();
      for (const [tid, t] of listed) {
        const c = content(t);
        score.set(tid, [...c].filter((w) => said.has(w)).length / Math.max(c.size, 1));
      }
      const best = Math.max(...score.values());
      if (best > 0) return score.get(taskId) < best ? `'${listed.get(taskId)}' is not the reminder the user named` : null;
    }
    return "no listed reminder matches the user's words";
  }
  function namesFile(msg, path) { // `path` occurs as a whole name ("payments.c" is not named by "payments.csv")
    const low = msg.toLowerCase(), p = path.toLowerCase();
    for (let i = low.indexOf(p); p && i !== -1; i = low.indexOf(p, i + 1)) {
      const after = low.slice(i + p.length, i + p.length + 1);
      if (!after || !(/[\p{L}\p{N}]/u.test(after) || after === "_")) return true;
    }
    return false;
  }
  function groundingIssue(action, events) {
    if (action.type !== "tool_call") return null;
    if (action.tool === "scheduler.delete") return deleteIssue(String(action.arguments.task_id || ""), events);
    const req = currentRequest(events), said = req.filter((e) => e.type === "user").map((e) => e.content);
    for (let i = 0; i + 1 < req.length; i++) { // repeating a call that failed for good can never help
      const prev = req[i], obs = req[i + 1];
      if (prev.type === "arouse" && obs.type === "tool_error" && !obs.content.retryable && prev.turn.action.tool === action.tool
        && canonicalJson(prev.turn.action.arguments || {}) === canonicalJson(action.arguments)) return `this exact call already failed: ${obs.content.error}`;
    }
    const field = COPY_FIELDS[action.tool];
    if (field && field in action.arguments && !copiedFrom(action.arguments[field], said)) return `${field} not grounded`;
    if (action.tool === "gst.calculate") { // an amount and a rate the user wrote (a rate is written with %)
      const [amounts, rates] = gstMentions(req);
      for (const [key, allowed] of [["amount", amounts], ["rate", rates]]) {
        const v = String(action.arguments[key] ?? "").trim(), low = new Set([...allowed].map((x) => x.toLowerCase()));
        if ((allowed.size && !low.has(v.toLowerCase())) || !v || !said.some((m) => m.toLowerCase().includes(v.toLowerCase())))
          return `${key} '${v}' is not the ${key} the user wrote`;
      }
    }
    if (action.tool === "scheduler.create" && "time" in action.arguments) {
      const times = mentionedTimes(said);
      if (times.size && !times.has(action.arguments.time)) return `time ${action.arguments.time} is not a time the user wrote`;
    }
    if (action.tool === "file.read") {
      const path = action.arguments.path || "";
      const listed = new Set(req.filter((e) => e.type === "tool_result").flatMap((e) => e.content.files || []));
      if (!listed.has(path) && !said.some((m) => namesFile(m, path))) return "path not grounded";
    }
    return null;
  }
  const ANSWER_WORD = /[a-z0-9][a-z0-9_.:'-]*[a-z0-9]|[a-z0-9]/gi;
  // "₹1,374.10" and the tool's 1374.1 must compare equal: drop digit grouping and trailing decimal zeros
  const normalizeNumbers = (t) => t.replace(/(?<=\d),(?=\d)/g, "").replace(/(\d+)\.(\d*?)0+(?!\d)/g, (_, a, b) => a + (b ? "." + b : ""));
  const answerWords = (t) => (normalizeNumbers(t).match(ANSWER_WORD) || []).map((w) => w.toLowerCase());
  function answerIssue(action, events, vocab) {
    const text = action.result || action.question || action.error;
    if (!text || !vocab || !vocab.size) return null;
    const sources = [];
    for (const ev of currentRequest(events)) {
      if (ev.type === "user") sources.push(ev.content);
      else if (ev.type === "tool_result" || ev.type === "tool_error") sources.push(pyJson(ev.content));
      else if (ev.type === "arouse" && ev.turn.action.type === "tool_call") sources.push(pyJson(ev.turn.action.arguments));
    }
    const seen = new Set(sources.flatMap(answerWords)), blob = normalizeNumbers(sources.join(" ")).toLowerCase();
    const unknown = answerWords(text).filter((w) => !vocab.has(w) && !seen.has(w) && !/^\d{1,3}(st|nd|rd|th)?$/.test(w)
      && !(/\d/.test(w) && blob.includes(w)));
    if (unknown.length) return `answer uses unknown words: ${unknown.slice(0, 5)}`;
    return claimIssue(action, events);
  }
  // A "couldn't find a reminder to X" answer must agree with the tools and name what the user asked for.
  const NOT_FOUND = /couldn't find a reminder to (.+?)\.?$/i;
  function claimIssue(action, events) {
    const m = action.type === "fail" ? NOT_FOUND.exec(action.error || "") : null;
    if (!m) return null;
    const target = m[1].trim(), req = currentRequest(events);
    const listed = new Set(req.filter((e) => e.type === "tool_result").flatMap((e) => (e.content.reminders || []).map((r) => r.task.toLowerCase())));
    if (listed.has(target.toLowerCase())) return `'${target}' was in the listed reminders`;
    const v = words(target).join("\u0000");
    const said = req.filter((e) => e.type === "user").map((e) => words(e.content));
    if (!said.some((w) => w.some((_, i) => w.slice(i, i + words(target).length).join("\u0000") === v))) return `'${target}' is not what the user asked for`;
    return null;
  }
  // json.dumps default formatting (", " and ": "), used only for word extraction
  function pyJson(v) { return JSON.stringify(v).replace(/","/g, '", "').replace(/":/g, '": '); }

  // ---------------------------------------------------------------- constrained decoding
  const MONTH_RE = [...new Set([...MONTHS, ...MONTHS.map((m) => m.slice(0, 3)), "sept"])].sort((a, b) => b.length - a.length).join("|");
  const DATE_PATTERNS = [
    new RegExp(`\\b(${MONTH_RE})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?\\b`, "gi"),
    new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?(${MONTH_RE})\\b`, "gi"),
  ];
  const DAYS_RE = "monday|tuesday|wednesday|thursday|friday|saturday|sunday";
  const STOP_ALL = /[?!;]|\.(?=\s|$)|\s(?:please|pls|thanks|thank you|thx)\b/i;
  const STOP_PATH = /[\s?!;,]|\.(?:\s|$)/;
  const STOP_TASK = new RegExp(`[,:]|\\s(?:at|in|on|by|after|from|before)\\s+(?:\\d|an?\\s|half|the\\s\\d|${DAYS_RE}|noon|midnight)` +
    `|\\s(?:tomorrow|today|tonight)\\b|\\s(?:every|each)\\s|\\s(?:${DAYS_RE})\\b` +
    "|\\s(?:daily|weekly|monthly)(?=\\s*(?:$|[.,!?;]|(?:at|on|in|from|starting|please|thanks)\\b))" +
    `|\\s(?:next|this)\\s+(?:${DAYS_RE}|week|weekend|month|year|morning|afternoon|evening|night)\\b`, "i");
  const OPEN_RE = /"(task|text|query|path|date|time|amount|rate)":"((?:[^"\\]|\\.)*)$/;
  const CHOICE_FIELDS = ["date", "time", "amount", "rate"]; // values chosen from a fixed list of candidates
  const CLOSE_RE = /^"([,}\]][\s\S]*)?$/;
  const isAlnum = (ch) => /[\p{L}\p{N}]/u.test(ch);

  function mentionedDates(texts, today) {
    const out = new Set();
    for (const text of texts) {
      DATE_PATTERNS.forEach((pat, i) => {
        for (const m of text.matchAll(pat)) {
          const [ms, ds] = i === 0 ? [m[1], m[2]] : [m[2], m[1]];
          const month = MONTHS.findIndex((n) => n.startsWith(ms.toLowerCase().slice(0, 3)));
          for (const year of [today.getFullYear(), today.getFullYear() + 1]) {
            const d = new Date(year, month, Number(ds));
            if (d.getMonth() !== month) break; // invalid day for this month
            if (fmtDate(d) >= fmtDate(today)) { out.add(fmtDate(d)); break; }
          }
        }
      });
    }
    return out;
  }

  function vocabPieces(tok) {
    if (tok._pieces) return tok._pieces;
    const pieces = new Map(), closing = [], closerText = new Map();
    for (let t = NUM_SPECIAL; t < tok.vocabSize; t++) {
      let p;
      try { p = decStrict.decode(tok.idBytes(t)); } catch (e) { continue; }
      if (CLOSE_RE.test(p)) { closing.push(t); closerText.set(t, p); }
      else if (p && !p.includes('"') && !p.includes("\\")) { if (!pieces.has(p)) pieces.set(p, []); pieces.get(p).push(t); }
    }
    let maxlen = 0;
    for (const k of pieces.keys()) maxlen = Math.max(maxlen, k.length);
    tok._pieces = { pieces, closing, closerText, maxlen };
    return tok._pieces;
  }

  const LABEL = /^\s*([^:\n]{1,60}?):\s+(?=\S)/;
  function labelEnd(src) { // "<short label>: <content>": where the content starts (up to two labels)
    let end = null;
    for (let k = 0; k < 2; k++) {
      const m = LABEL.exec(src.slice(end || 0));
      if (!m || m[1].split(/\s+/).filter(Boolean).length > 6) break;
      end = (end || 0) + m[0].length;
    }
    return end;
  }

  function referencedDates(texts, now) {
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()), out = new Set();
    for (const text of texts) {
      const low = text.toLowerCase(), ws = new Set(low.match(/[a-z]+/g) || []);
      if (ws.has("today") || ws.has("tonight")) out.add(fmtDate(today));
      if (ws.has("tomorrow")) out.add(fmtDate(addDays(today, low.includes("after tomorrow") ? 2 : 1)));
      WEEKDAYS.forEach((name, w) => { if (ws.has(name)) out.add(fmtDate(addDays(today, (w - pyWeekday(today) + 7) % 7 || 7))); });
    }
    for (const d of mentionedDates(texts, today)) out.add(d);
    return out;
  }

  class CopyConstraint {
    constructor(tok, events, context) {
      this.tok = tok;
      Object.assign(this, vocabPieces(tok));
      const req = currentRequest(events);
      const said = req.filter((e) => e.type === "user").map((e) => e.content);
      const listed = req.filter((e) => e.type === "tool_result").flatMap((e) => e.content.files || []);
      this.sources = { task: said, text: said, query: said, path: said.concat(listed) };
      this.listed = new Set(listed);
      let dates = new Set();
      if (context && context.now) {
        dates.add(context.now.slice(0, 10));
        for (const m of String(context.calendar || "").matchAll(/\d{4}-\d{2}-\d{2}/g)) dates.add(m[0]);
        const asked = req.filter((e) => e.type === "arouse" && e.turn.action.type === "ask_user").map((e) => e.turn.action.question);
        const [y, mo, d] = context.now.slice(0, 10).split("-").map(Number);
        const ref = referencedDates(said.concat(asked), new Date(y, mo - 1, d));
        if (ref.size) dates = ref;
      }
      this.dates = [...dates].sort();
      const [amounts, rates] = gstMentions(req);
      this.pathSpans = new Set();
      for (const src of said) { // "vet_visits.txt" in "Count the lines in vet_visits.txt." (never "vet_visits.")
        for (let i = 0; i < src.length; i++) {
          if (isAlnum(src[i]) && (i === 0 || !(isAlnum(src[i - 1]) || "_-./\\".includes(src[i - 1])))) {
            const m = STOP_PATH.exec(src.slice(i)), span = m ? src.slice(i, i + m.index) : src.slice(i);
            if (span.includes(".")) this.pathSpans.add(span);
          }
        }
      }
      this.choices = { date: this.dates, time: [...mentionedTimes(said)].sort(), amount: [...amounts].sort(), rate: [...rates].sort() };
    }
    openField(generated) {
      const at = generated.lastIndexOf(S.TOOL_CALL);
      if (at < 0) return null;
      const body = this.tok.decode(generated.slice(at + 1)).replace(/\uFFFD/g, ""); // like Python's errors="ignore"
      const m = OPEN_RE.exec(body);
      if (!m || m[2].includes("\\")) return null;
      return [m[1], m[2], body];
    }
    nextAfterClose(field, body) {
      if (field === "date" || field === "amount") return ",";
      if (field === "task" && body.includes('"scheduler.create"') && !body.includes('"in_minutes"')) return ",";
      return "}";
    }
    continuations(field, partial) {
      if (CHOICE_FIELDS.includes(field)) return this.choices[field].filter((d) => d.startsWith(partial) && d !== partial).map((d) => d.slice(partial.length));
      const out = [];
      const word = field === "path" ? (ch) => isAlnum(ch) || "_-./\\".includes(ch) : isAlnum;
      for (const src of this.sources[field]) {
        const whole = field === "path" && this.listed.has(src); // a listed file name is copied whole
        const pats = { task: [STOP_ALL, STOP_TASK], text: [STOP_ALL], query: [STOP_ALL], path: whole ? [] : [STOP_PATH] }[field];
        let starts = [];
        if (!partial) {
          for (let i = 0; i < src.length; i++) if (isAlnum(src[i]) && (i === 0 || !word(src[i - 1]))) starts.push(i);
        } else {
          for (let i = src.indexOf(partial); i !== -1; i = src.indexOf(partial, i + 1)) if (i === 0 || !word(src[i - 1])) starts.push(i);
        }
        if (whole) starts = starts.filter((i) => i === 0);
        const lab = field === "text" || field === "query" ? labelEnd(src) : null;
        if (lab !== null) starts = starts.filter((i) => i === lab); // a labelled note is the whole content after the label
        if (field === "task" && !partial) starts = starts.filter((i) => !/\d/.test(src[i]));
        for (const i of starts) {
          let rest = src.slice(i + partial.length);
          const stops = pats.map((p) => { const m = p.exec(src.slice(i)); return m ? m.index : -1; }).filter((x) => x >= 0);
          if (stops.length) {
            const cut = Math.min(...stops) - partial.length;
            if (cut <= 0) continue;
            rest = rest.slice(0, cut);
          }
          if (field === "path" && !this.complete(field, partial + rest)) continue; // whole file names only
          out.push(rest);
        }
      }
      return out;
    }
    complete(field, value) {
      if (CHOICE_FIELDS.includes(field)) return this.choices[field].includes(value);
      if (!value.trim() || value !== value.trim()) return false;
      if (field === "path") return this.listed.has(value) || this.pathSpans.has(value); // a whole file name
      if (field === "text" || field === "query") { // notes and queries are copied verbatim to the end of the sentence
        return this.sources[field].some((src) => {
          for (let i = src.indexOf(value); i !== -1; i = src.indexOf(value, i + 1)) {
            if (i > 0 && isAlnum(src[i - 1])) continue;
            const lab = labelEnd(src);
            if (lab !== null && lab !== i) continue;
            const rest = src.slice(i + value.length), m = STOP_ALL.exec(rest), before = m ? rest.slice(0, m.index) : rest;
            if (![...before].some(isAlnum)) return true;
          }
          return false;
        });
      }
      return copiedFrom(value, this.sources[field]);
    }
    hook() {
      return (generated, logits) => {
        const st = this.openField(generated);
        if (!st) return logits;
        const [field, partial, body] = st;
        if (CHOICE_FIELDS.includes(field) ? !this.choices[field].length : !this.sources[field].length) return logits;
        const allowed = new Set();
        for (const rest of this.continuations(field, partial)) {
          for (let n = 1; n <= Math.min(rest.length, this.maxlen); n++) {
            const ids = this.pieces.get(rest.slice(0, n));
            if (ids) ids.forEach((t) => allowed.add(t));
          }
        }
        if (this.complete(field, partial)) {
          const need = this.nextAfterClose(field, body);
          let closers = this.closing.filter((t) => this.closerText.get(t).charAt(1) === need);
          if (!closers.length) closers = this.closing.filter((t) => this.closerText.get(t) === '"');
          closers.forEach((t) => allowed.add(t));
        }
        const ok = [...allowed].filter((t) => Number.isFinite(logits[t]));
        if (!ok.length) return logits;
        const out = new Float32Array(logits.length).fill(-Infinity);
        for (const t of ok) out[t] = logits[t];
        return out;
      };
    }
  }

  const err = (message, retryable = false) => ({ success: false, error: message, retryable });

  // ---------------------------------------------------------------- knowledge base (arouse/agent/knowledge.py)
  let KB = null;
  function setKnowledge(kb) {
    const S = kb.search, stop = new Set(S.stopwords), noStem = new Set(S.no_stem), syn = S.synonyms;
    const stem = (w0) => {
      let w = syn[w0] || w0;
      if (noStem.has(w) || /^[0-9]+$/.test(w)) return w;
      if (w.length > 4 && w.endsWith("ies")) w = w.slice(0, -3) + "y";
      else if (w.length > 3 && w.endsWith("xes")) w = w.slice(0, -2);
      else if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) w = w.slice(0, -1);
      if (w.length > 5 && w.endsWith("ing")) w = w.slice(0, -3);
      else if (w.length > 4 && w.endsWith("ed")) w = w.slice(0, -2);
      if (w.length > 3 && w.endsWith("e")) w = w.slice(0, -1);
      return w;
    };
    const terms = (text) => {
      const out = [];
      for (const w of text.toLowerCase().replace(/\be[- ](?=invoic|way)/g, "e").match(/[a-z]+|[0-9]+/g) || []) {
        if ((w.length > 1 || /^[0-9]+$/.test(w)) && !stop.has(w)) { const t = stem(w); if (!out.includes(t)) out.push(t); }
      }
      return out;
    };
    const entries = kb.entries;
    const phrasings = entries.map((e) => [e.q, ...e.alts].map((p) => new Set(terms(p))));
    const strong = phrasings.map((ps) => new Set(ps.flatMap((x) => [...x])));
    const weak = entries.map((e) => new Set(terms(e.a)));
    const df = new Map();
    strong.forEach((st, i) => { for (const t of new Set([...st, ...weak[i]])) df.set(t, (df.get(t) || 0) + 1); });
    const n = entries.length, idf = new Map([...df].map(([t, c]) => [t, Math.log(1 + n / c)]));
    KB = { kb, terms, stem, entries, phrasings, strong, weak, idf, n, leadStop: new Set(S.lead_stopwords) };
  }
  function kbSearch(query) {
    if (!KB) return { found: false };
    const { idf } = KB, q = KB.terms(query), known = q.filter((t) => idf.has(t));
    if (!known.length || q.length - known.length >= known.length) return { found: false };
    let knownW = 0;
    for (const t of known) knownW += idf.get(t);
    let best = -1, bestScore = 0, bestCover = 0;
    KB.strong.forEach((st, i) => {
      if (!known.some((t) => st.has(t))) return;
      const wk = KB.weak[i];
      let num = 0;
      for (const t of known) num += idf.get(t) * (st.has(t) ? 1 : wk.has(t) ? 0.5 : 0);
      const cover = num / knownW;
      let prec = 0;
      for (const ph of KB.phrasings[i]) {
        let denom = 0, shared = 0;
        for (const t of [...ph].sort()) denom += idf.get(t);
        for (const t of known) if (ph.has(t)) shared += idf.get(t);
        if (denom) prec = Math.max(prec, shared / denom);
      }
      const score = cover + 0.25 * prec;
      if (score > bestScore + 1e-9) { best = i; bestScore = score; bestCover = cover; }
    });
    if (best < 0 || bestCover < 0.5) return { found: false };
    const e = KB.entries[best];
    return { found: true, id: e.id, question: e.q, answer: e.a };
  }

  // ---------------------------------------------------------------- GST and leads (arouse/agent/business.py)
  const AMOUNT_RE = /^(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|thousand|lakhs?|lacs?|crores?|cr)?\s*(?:rupees|rs\.?|\/-)?$/i;
  const RATE_RE = /^([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent|per cent)?$/i;
  const UNIT = { k: 1e3, thousand: 1e3, lakh: 1e5, lakhs: 1e5, lac: 1e5, lacs: 1e5, crore: 1e7, crores: 1e7, cr: 1e7 };
  function decimalToInt(num, scale) {
    const [whole, frac = ""] = num.split("."), digits = String(scale).length - 1;
    const f = digits ? (frac + "0".repeat(digits)).slice(0, digits) : "";
    return Number(whole || "0") * scale + (f ? Number(f) : 0);
  }
  function parseAmount(text) {
    const m = AMOUNT_RE.exec(text.trim());
    if (!m) return null;
    const unit = UNIT[(m[2] || "").toLowerCase()] || 1, digits = m[1].replace(/,/g, "");
    const paise = m[1].includes(".") ? decimalToInt(digits, 100 * unit) : Number(digits) * 100 * unit;
    return paise > 0 && paise <= 1e15 ? paise : null;
  }
  function parseRate(text) {
    const m = RATE_RE.exec(text.trim());
    if (!m) return null;
    const bp = decimalToInt(m[1], 100);
    return bp >= 0 && bp <= 10000 ? bp : null;
  }
  const money = (paise) => paise / 100;
  const toPaise = (x) => Math.round(x * 100);
  function fmtInr(paise) {
    const rupees = Math.floor(paise / 100), p = paise % 100;
    let s = String(rupees);
    if (s.length > 3) {
      let head = s.slice(0, -3);
      const tail = s.slice(-3), groups = [];
      while (head.length > 2) { groups.unshift(head.slice(-2)); head = head.slice(0, -2); }
      s = (head ? [head, ...groups, tail] : [...groups, tail]).join(",");
    }
    return s + (p ? "." + String(p).padStart(2, "0") : "");
  }
  function fmtRate(bp) {
    const whole = Math.floor(bp / 100), frac = bp % 100;
    return frac ? `${whole}.${String(frac).padStart(2, "0")}`.replace(/0+$/, "") : String(whole);
  }
  function gstCalculate(amount, rate, inclusive = false) {
    const a0 = parseAmount(amount), r0 = parseRate(rate);
    if (a0 === null) return [false, err(`I couldn't read the amount '${amount}'`)];
    if (r0 === null) return [false, err(`I couldn't read the GST rate '${rate}'`)];
    const a = BigInt(a0), r = BigInt(r0);
    let base, gst;
    if (inclusive) { const d = 10000n + r; base = (a * 10000n * 2n + d) / (2n * d); gst = a - base; }
    else { base = a; gst = (a * r + 5000n) / 10000n; }
    const cgst = (gst + 1n) / 2n, N = Number;
    return [true, { success: true, amount: money(a0), rate: money(r0), inclusive: Boolean(inclusive), taxable_value: money(N(base)),
      gst: money(N(gst)), cgst: money(N(cgst)), sgst: money(N(gst - cgst)), total: money(N(base + gst)) }];
  }

  const CUSTOMER_FILE = "customers.csv", METHODS = ["inactive", "occasions", "top", "custom"];
  function parseCsv(text) {
    const rows = [];
    let row = [], field = "", quoted = false;
    const endField = () => { row.push(field.trim()); field = ""; };
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (c === '"') quoted = false;
        else field += c;
      } else if (c === '"') quoted = true;
      else if (c === ",") endField();
      else if (c === "\r" || c === "\n") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        endField();
        if (row.some(Boolean)) rows.push(row);
        row = [];
      } else field += c;
    }
    endField();
    if (row.some(Boolean)) rows.push(row);
    return rows;
  }
  const DAY = 864e5;
  const dayIso = (n) => new Date(n * DAY).toISOString().slice(0, 10);
  function isoDay(s) {
    const m = /^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(s.trim().slice(0, 10));
    if (!m) return null;
    const y = +m[1], mo = +m[2], d = +m[3], t = new Date(Date.UTC(y, mo - 1, d));
    return t.getUTCMonth() === mo - 1 && t.getUTCDate() === d ? Date.UTC(y, mo - 1, d) / DAY : null;
  }
  function monthDay(s) {
    const m = /(\d{1,2})-(\d{1,2})$/.exec(s.trim());
    if (!m) return null;
    const mo = +m[1], d = +m[2];
    return mo >= 1 && mo <= 12 && d >= 1 && d <= 31 ? [mo, d] : null;
  }
  function nextOccurrence([mo, d0], today) {
    const year0 = new Date(today * DAY).getUTCFullYear();
    let occ;
    for (const year of [year0, year0 + 1]) {
      let d = d0;
      for (;;) { const t = new Date(Date.UTC(year, mo - 1, d)); if (t.getUTCMonth() === mo - 1) { occ = Date.UTC(year, mo - 1, d) / DAY; break; } d--; }
      if (occ >= today) return occ;
    }
    return occ;
  }
  const spent = (row) => parseAmount(row.total_spent || "0") || 0;
  function compareKeys(a, b) {
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
  }
  function customKeywords(query) { return KB ? KB.terms(query).filter((t) => !KB.leadStop.has(t)) : []; }
  function findLeads(files, now, method, query) {
    if (!METHODS.includes(method)) return [false, err(`unknown method ${method}`)];
    if ((method === "custom") !== (query !== undefined && query !== null)) return [false, err("method custom needs a query; the other methods take none")];
    if (!(CUSTOMER_FILE in files)) return [false, err(`file not found: ${CUSTOMER_FILE}`)];
    const rows = parseCsv(files[CUSTOMER_FILE]);
    if (!rows.length) return [false, err(`${CUSTOMER_FILE} is empty`)];
    const head = rows[0].map((h) => h.trim().toLowerCase());
    const need = { inactive: "last_purchase", occasions: null, top: "total_spent", custom: null }[method];
    if (!head.includes("name") || (need && !head.includes(need))) return [false, err(`${CUSTOMER_FILE} needs the columns name and ${need || "city"}`)];
    if (method === "occasions" && !head.includes("birthday") && !head.includes("anniversary")) return [false, err(`${CUSTOMER_FILE} needs a birthday or anniversary column`)];
    const people = rows.slice(1).map((r) => Object.fromEntries(head.map((h, i) => [h, i < r.length ? r[i] : ""])));
    const today = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) / DAY, found = [];
    if (method === "inactive") {
      for (const p of people) { const last = isoDay(p.last_purchase || ""); if (last !== null && today - last >= 90) found.push([[-spent(p), p.name], p, `last bought ${dayIso(last)}`]); }
    } else if (method === "occasions") {
      for (const p of people) {
        let best = null;
        for (const kind of ["birthday", "anniversary"]) {
          const md = monthDay(p[kind] || "");
          if (!md) continue;
          const occ = nextOccurrence(md, today), days = occ - today;
          if (days <= 30 && (best === null || days < best[0])) best = [days, `${kind} on ${dayIso(occ)}`];
        }
        if (best) found.push([[best[0], p.name], p, best[1]]);
      }
    } else if (method === "top") {
      for (const p of people) if (spent(p) > 0) found.push([[-spent(p), p.name], p, `spent ₹${fmtInr(spent(p))}`]);
    } else {
      const keys = customKeywords(query || "");
      for (const p of people) {
        const have = new Set(KB.terms([p.name || "", p.city || "", p.interest || ""].join(" ")));
        if (keys.length && keys.every((k) => have.has(k))) found.push([[-spent(p), p.name], p, p.interest ? `likes ${p.interest}` : "matches your idea"]);
      }
    }
    found.sort((x, y) => compareKeys(x[0], y[0]));
    const leads = found.slice(0, 3).map(([, p, why]) => ({ name: p.name, city: p.city || "", why }));
    const out = { success: true, method, count: found.length, leads };
    if (method === "custom") out.query = query;
    return [true, out];
  }

  // ---------------------------------------------------------------- sandbox tools

  function firstRun(now, hh, mm, repeat) {
    let d = new Date(now.getFullYear(), now.getMonth(), now.getDate(), hh, mm, 0, 0);
    for (let k = 0; k < 400; k++) {
      if (d > now) {
        if (repeat.freq === "daily") return d;
        if (repeat.freq === "weekly" && repeat.by_day.includes(DAY_CODES[pyWeekday(d)])) return d;
        if (repeat.freq === "monthly" && repeat.by_month_day.includes(d.getDate())) return d;
      }
      d = addDays(d, 1);
    }
    throw new ProtocolError("repeat rule never matches");
  }

  class Sandbox {
    constructor(state = {}) {
      this.reminders = (state.reminders || []).map((r) => ({ ...r }));
      this.notes = (state.notes || []).map((n) => ({ ...n }));
      this.files = { ...(state.files || {}) };
      this.nextId = state.nextId || 1 + Math.max(0, ...this.reminders.map((r) => Number(r.task_id.split("-")[1]) || 0));
      this.nextNote = state.nextNote || this.notes.length + 1;
      this.failures = { ...(state.failures || {}) }; // upcoming transient failures per tool (tests)
      this.now = () => { const d = new Date(); d.setSeconds(0, 0); return d; };
    }
    snapshot() {
      const rem = [...this.reminders].sort((a, b) => (a.next_run < b.next_run ? -1 : a.next_run > b.next_run ? 1 : a.task_id < b.task_id ? -1 : 1));
      return { reminders: rem.map((r) => ({ ...r })), notes: this.notes.map((n) => ({ ...n })) };
    }
    state() { return { ...this.snapshot(), files: { ...this.files }, nextId: this.nextId, nextNote: this.nextNote }; }
    execute(tool, args) {
      try { validateCall(tool, args); } catch (e) { return [false, err(`invalid arguments: ${e.message}`)]; }
      if (this.failures[tool] > 0) { this.failures[tool] -= 1; return [false, err(`${tool.split(".")[0]} unavailable`, true)]; }
      return this["_" + tool.replace(".", "_")](args);
    }
    _scheduler_create({ task, date, time, in_minutes, repeat }) {
      const now = this.now();
      let when;
      if (in_minutes !== undefined) {
        if (date || time || repeat) return [false, err("invalid arguments: in_minutes cannot be combined with date/time/repeat")];
        when = new Date(now.getTime() + in_minutes * 60000);
      } else if (repeat !== undefined) {
        if (date || !time) return [false, err("invalid arguments: a recurring reminder needs time and repeat (no date)")];
        const f = repeat.freq;
        if ((f === "weekly") !== ("by_day" in repeat) || (f === "monthly") !== ("by_month_day" in repeat))
          return [false, err("invalid arguments: weekly needs by_day, monthly needs by_month_day")];
        const [hh, mm] = time.split(":").map(Number);
        when = firstRun(now, hh, mm, repeat);
      } else if (time && !date) {
        const [hh, mm] = time.split(":").map(Number);
        when = firstRun(now, hh, mm, { freq: "daily" });
      } else {
        if (!(date && time)) return [false, err("invalid arguments: need date and time, in_minutes, or time and repeat")];
        const [y, mo, d] = date.split("-").map(Number), [hh, mm] = time.split(":").map(Number);
        when = new Date(y, mo - 1, d, hh, mm);
        if (when.getMonth() !== mo - 1 || when.getDate() !== d) return [false, err(`invalid date ${date}`)];
        if (when <= now) return [false, err("that time has already passed")];
      }
      const rid = `r-${this.nextId++}`;
      const r = { task_id: rid, task, next_run: fmtDT(when) };
      if (repeat) r.repeat = repeat;
      this.reminders.push(r);
      return [true, { success: true, task_id: rid, task, next_run: r.next_run }];
    }
    _scheduler_list() { const items = this.snapshot().reminders; return [true, { success: true, count: items.length, reminders: items }]; }
    _scheduler_delete({ task_id }) {
      const i = this.reminders.findIndex((r) => r.task_id === task_id);
      if (i < 0) return [false, err(`no reminder with task_id ${task_id}`)];
      const [r] = this.reminders.splice(i, 1);
      return [true, { success: true, deleted: task_id, task: r.task }];
    }
    _notes_create({ text }) {
      const nid = `n-${this.nextNote++}`;
      this.notes.push({ note_id: nid, text });
      return [true, { success: true, note_id: nid }];
    }
    _file_read({ path }) {
      if (!(path in this.files)) return [false, err(`file not found: ${path}`)];
      const lines = this.files[path].split(/\r\n|\r|\n/);
      if (lines.length && lines[lines.length - 1] === "") lines.pop();
      return [true, { success: true, path, lines: lines.length, preview: lines.slice(0, 2).join("\n").slice(0, 120) }];
    }
    _file_list() { return [true, { success: true, files: Object.keys(this.files).sort() }]; }
    _kb_search({ query }) { return [true, { success: true, ...kbSearch(query) }]; }
    _gst_calculate({ amount, rate, inclusive }) { return gstCalculate(amount, rate, inclusive || false); }
    _leads_find({ method, query }) { return findLeads(this.files, this.now(), method, query); }
    /** Reminders whose time has come: one-time ones are removed, recurring ones move to their next run. */
    due(now) {
      const fired = [];
      for (const r of [...this.reminders]) {
        if (r.next_run > fmtDT(now)) continue;
        fired.push({ ...r });
        if (!r.repeat) { this.reminders.splice(this.reminders.indexOf(r), 1); continue; }
        const [hh, mm] = r.next_run.slice(11).split(":").map(Number);
        let nxt = firstRun(now, hh, mm, r.repeat);
        if (r.repeat.freq === "daily" && (r.repeat.interval || 1) > 1) {
          const [y, mo, d] = r.next_run.slice(0, 10).split("-").map(Number);
          nxt = new Date(y, mo - 1, d, hh, mm);
          while (nxt <= now) nxt = addDays(nxt, r.repeat.interval);
        }
        r.next_run = fmtDT(nxt);
      }
      return fired;
    }
  }

  // ---------------------------------------------------------------- reply formats (arouse/agent/answers.py)
  const ordinal = (n) => `${n}${n % 100 >= 10 && n % 100 <= 20 ? "th" : { 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th"}`;
  function ruleWords(r) {
    if (r.freq === "daily") { const n = r.interval || 1; return n === 1 ? "every day" : `every ${n} days`; }
    if (r.freq === "monthly") return `on the ${ordinal(r.by_month_day[0])} of every month`;
    const days = r.by_day;
    if (days.join() === DAY_CODES.slice(0, 5).join()) return "every weekday";
    if (days.join() === "SA,SU") return "every weekend";
    return "every " + days.map((d) => cap(WEEKDAYS[DAY_CODES.indexOf(d)])).join(" and ");
  }
  function createdMsg(res, repeat) {
    const [date, time] = res.next_run.split("T");
    return `Reminder set: ${res.task} on ${date} at ${time}` + (repeat ? `, repeating ${ruleWords(repeat)}.` : ".");
  }
  function listMsg(items) {
    if (!items.length) return "You have no reminders.";
    const parts = items.map((x) => `${x.task} on ${x.next_run.replace("T", " at ")}` + ("repeat" in x ? " (repeats)" : ""));
    return `You have ${items.length} reminder${items.length > 1 ? "s" : ""}: ${parts.join("; ")}.`;
  }
  const R_UNKNOWN = "I don't know that yet. I can answer questions about GST, gold and jewellery, dairy and running a small business.";
  const LEAD_DESC = { inactive: "no purchase in 3 months", occasions: "birthday or anniversary in 30 days",
    top: "top customers, ask them for referrals", custom: "your idea" };
  function gstMsg(res) {
    const r = fmtRate(toPaise(res.rate)), [a, g, c, sg] = ["amount", "gst", "cgst", "sgst"].map((k) => fmtInr(toPaise(res[k])));
    if (res.inclusive) return `₹${a} includes ₹${g} GST at ${r}% (CGST ₹${c} + SGST ₹${sg}). Price before GST: ₹${fmtInr(toPaise(res.taxable_value))}.`;
    return `GST at ${r}% on ₹${a} is ₹${g} (CGST ₹${c} + SGST ₹${sg}). Total: ₹${fmtInr(toPaise(res.total))}.`;
  }
  function leadsMsg(res) {
    const desc = LEAD_DESC[res.method], n = res.count, leads = res.leads;
    if (n === 0) return res.method === "custom" ? `I didn't find any customers matching "${res.query}" in customers.csv.`
      : `I didn't find any leads (${desc}) in customers.csv.`;
    const items = leads.map((x) => (x.city ? `${x.name} (${x.city}, ${x.why})` : `${x.name} (${x.why})`)).join("; ");
    return `Found ${n} lead${n !== 1 ? "s" : ""} (${desc}): ${items}.${n > leads.length ? ` Showing ${leads.length}.` : ""}`;
  }
  function answerFromResult(events) {
    for (let i = events.length - 1; i > 0; i--) {
      const ev = events[i];
      if (ev.type === "user" || ev.type === "tool_error") return null;
      if (ev.type !== "tool_result") continue;
      const prev = events[i - 1];
      if (prev.type !== "arouse" || prev.turn.action.type !== "tool_call") return null;
      const { tool, arguments: args } = prev.turn.action, res = ev.content;
      try {
        if (tool === "scheduler.create") return createdMsg(res, args.repeat);
        if (tool === "scheduler.list") return listMsg(res.reminders);
        if (tool === "scheduler.delete") return `Deleted the reminder: ${res.task}.`;
        if (tool === "notes.create") return "Saved the note.";
        if (tool === "file.read") return `${res.path} has ${res.lines} lines. It starts with: ${res.preview.split("\n")[0]}`;
        if (tool === "file.list") return `You have ${res.files.length} files: ${res.files.join(", ")}.`;
        if (tool === "kb.search") return res.found ? res.answer : R_UNKNOWN;
        if (tool === "gst.calculate") return gstMsg(res);
        if (tool === "leads.find") return leadsMsg(res);
      } catch (e) { return null; }
      return null;
    }
    return null;
  }

  /** A finish confirming a successful tool call must state what the tool returned (see answers.py). */
  function correctedConfirmation(action, events) {
    if (action.type !== "finish") return null;
    const expected = answerFromResult(events), text = action.result || "";
    if (expected === null || text === expected || (expected.startsWith(text) && text.endsWith("."))) return null;
    return expected;
  }

  /** The "which file?" question after a failed file.read and a file.list, stated from those results. */
  function correctedQuestion(action, events) {
    if (action.type !== "ask_user" || !(action.question || "").startsWith("I couldn't find")) return null;
    let files = null, path = null;
    for (let i = events.length - 1; i > 0; i--) {
      const ev = events[i];
      if (ev.type === "user") break;
      const prev = events[i - 1], call = prev.type === "arouse" ? prev.turn.action : {};
      if (ev.type === "tool_result" && call.tool === "file.list" && files === null) files = ev.content.files || [];
      if (ev.type === "tool_error" && call.tool === "file.read" && files !== null) { path = call.arguments.path; break; }
    }
    if (files === null || !path) return null;
    const stem = path.includes(".") ? path.slice(0, path.lastIndexOf(".")) : path, cands = files.filter((f) => f.startsWith(stem + "_"));
    const expected = cands.length ? `I couldn't find ${path}. Did you mean ${cands[0]}?`
      : `I couldn't find ${path}. Which file should I use? Available: ${files.join(", ")}.`;
    return action.question === expected ? null : expected;
  }

  // ---------------------------------------------------------------- runtime
  function lastObservation(events) {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].type === "user") return null;
      if (events[i].type === "tool_result" || events[i].type === "tool_error") return events[i].type;
    }
    return null;
  }

  class Runtime {
    constructor(engine, { vocab = new Set(), maxToolCalls = 6, maxNew = 200, retries = 2, groundingSamples = 4,
      guardCompletion = true, constrainCopy = true } = {}) {
      Object.assign(this, { engine, vocab, maxToolCalls, maxNew, retries, groundingSamples, guardCompletion, constrainCopy });
    }
    fit(header, events) {
      const tok = this.engine.tok, ctx = this.engine.contextLength, budget = ctx - this.maxNew;
      let evs = events.slice();
      for (;;) {
        const ids = encodePrompt(tok, header, evs);
        if (ids.length <= budget) return ids;
        const nxt = evs.findIndex((e, i) => i > 0 && e.type === "user");
        if (nxt < 0) {
          if (ids.length <= ctx - 48) return ids;
          throw new ProtocolError("conversation too long");
        }
        evs = evs.slice(nxt);
      }
    }
    sample(prompt, attempt, hook) {
      const g = this.engine.generate(prompt, { maxNew: this.maxNew, temperature: attempt === 0 ? 0 : 0.7, topP: 0.95,
        seed: attempt, stop: [S.END], banned: [S.EOS], hook });
      try {
        const turn = decodeTurn(this.engine.tok, g.ids);
        if (turn.action.type === "tool_call") validateCall(turn.action.tool, turn.action.arguments);
        return [turn, g.text];
      } catch (e) {
        if (e instanceof ProtocolError) return [null, g.text];
        throw e;
      }
    }
    nextTurn(header, events) {
      const prompt = this.fit(header, events);
      const hook = this.constrainCopy ? new CopyConstraint(this.engine.tok, events, header.context).hook() : null;
      let raw = "", firstValid = null, attempt = 0;
      while (attempt < 1 + this.retries + (firstValid ? this.groundingSamples : 0)) {
        const [turn, text] = this.sample(prompt, attempt, hook);
        raw = text; attempt += 1;
        if (!turn) continue;
        const res = { turn, attempts: attempt, raw, valid: true, guarded: false, grounded: true };
        if (this.retries === 0 || !(groundingIssue(turn.action, events) || answerIssue(turn.action, events, this.vocab))) return this.guard(res, events);
        if (!firstValid) firstValid = { ...res, grounded: false };
      }
      if (firstValid) {
        let r = { ...firstValid, attempts: attempt };
        const a = r.turn.action, fixed = a.type === "finish" && answerIssue(a, events, this.vocab) ? answerFromResult(events) : null;
        if (fixed) r = { ...r, rewritten: true, turn: { action: { type: "finish", result: fixed }, plan: r.turn.plan,
          verify: "runtime: answer written from the tool result" } };
        else if (a.type === "tool_call" && a.tool === "scheduler.delete") { // never delete a reminder the user did not name
          r = { ...r, rewritten: true, turn: { action: { type: "fail",
            error: "I couldn't find a reminder that matches what you asked, so I didn't delete anything." }, plan: r.turn.plan,
          verify: `runtime: ${groundingIssue(a, events)}` } };
        }
        return this.guard(r, events);
      }
      return { turn: { action: { type: "fail", error: "Sorry, I couldn't work out a valid next step for that request." }, plan: null,
        verify: "runtime: the model did not produce a valid action" }, attempts: attempt, raw, valid: false, guarded: false, grounded: true };
    }
    guard(res, events) {
      const fixed = this.retries ? correctedConfirmation(res.turn.action, events) : null; // system mode only
      if (fixed) res = { ...res, rewritten: true, turn: { action: { type: "finish", result: fixed }, plan: res.turn.plan,
        verify: "runtime: confirmation written from the tool result" } };
      const question = this.retries ? correctedQuestion(res.turn.action, events) : null;
      if (question) res = { ...res, rewritten: true, turn: { action: { type: "ask_user", question }, plan: res.turn.plan,
        verify: "runtime: question written from the tool results" } };
      if (this.guardCompletion && res.turn.action.type === "finish" && lastObservation(events) === "tool_error") {
        return { ...res, guarded: true, turn: { action: { type: "fail", error: "The last step failed, so the task was not completed." },
          plan: res.turn.plan, verify: "runtime guard: finish after a failed tool call" } };
      }
      return res;
    }
    /** Run until ask_user / finish / fail. `onStep` receives each new event as it happens. */
    run(header, events, execute, onStep = () => {}) {
      const history = events.slice(), added = [], turns = [];
      let calls = 0;
      for (;;) {
        let r;
        try { r = this.nextTurn(header, history); }
        catch (e) {
          if (!(e instanceof ProtocolError) || !turns.length) throw e;
          r = { turn: { action: { type: "fail", error: "This conversation is too long for me to continue. Please start a new chat." },
            plan: null, verify: "runtime: context full" }, valid: true, guarded: false };
        }
        turns.push(r);
        const ev = { type: "arouse", turn: r.turn };
        history.push(ev); added.push(ev); onStep(ev);
        const a = r.turn.action;
        if (a.type !== "tool_call") return { events: added, final: a, turns };
        if (calls >= this.maxToolCalls) {
          const stop = { type: "arouse", turn: { action: { type: "fail", error: "I stopped because the task needed too many steps." }, plan: null, verify: "runtime: step limit" } };
          history.push(stop); added.push(stop); onStep(stop);
          return { events: added, final: stop.turn.action, turns };
        }
        calls += 1;
        const [ok, payload] = execute(a.tool, a.arguments);
        const obs = { type: ok ? "tool_result" : "tool_error", content: payload };
        history.push(obs); added.push(obs); onStep(obs);
      }
    }
  }

  async function loadArouse(base = "model/", onProgress = () => {}) {
    const [cfg, tokSpec, vocabText, kb] = await Promise.all([
      fetch(base + "config.json").then((r) => r.json()),
      fetch(base + "tokenizer.json").then((r) => r.json()),
      fetch(base + "response_vocab.txt").then((r) => r.text()),
      fetch(base + "knowledge.json").then((r) => r.json()),
    ]);
    setKnowledge(kb);
    // weights.bin (raw float16), or base64 text where a host only serves text files
    const resp = await fetch(base + (cfg.weights_file || "weights.bin"));
    if (!resp.ok) throw new Error(`weights: HTTP ${resp.status}`);
    const total = Number(resp.headers.get("content-length")) || 0;
    let buffer;
    if (resp.body && total) {
      const reader = resp.body.getReader(), buf = new Uint8Array(total);
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf.set(value, got); got += value.length;
        onProgress(got / total);
      }
      buffer = buf.buffer;
    } else {
      buffer = await resp.arrayBuffer();
      onProgress(1);
    }
    if (cfg.weights_encoding === "base64") {
      const bin = atob(new TextDecoder().decode(buffer).replace(/\s+/g, "")), bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      buffer = bytes.buffer;
    }
    const tok = new Tokenizer(tokSpec), model = new Model(cfg, buffer), engine = new Engine(model, tok);
    const vocab = new Set(vocabText.split(/\s+/).filter(Boolean));
    return { cfg, tok, model, engine, runtime: new Runtime(engine, { vocab }) };
  }

  const api = { S, Tokenizer, Model, Engine, Runtime, Sandbox, CopyConstraint, ProtocolError, loadArouse, buildContext,
    calendarText, encodePrompt, encodeTurnBody, decodeTurn, canonicalJson, validateCall, groundingIssue, answerIssue,
    copiedFrom, mentionedDates, answerFromResult, correctedConfirmation, setKnowledge, kbSearch, gstCalculate, findLeads,
    parseAmount, parseRate, fmtInr, customKeywords, labelEnd, mentionedTimes, gstMentions, correctedQuestion, TOOL_NAMES, SYSTEM_PROMPT, fmtDT, firstRun };
  root.Arouse = api;
})(typeof self !== "undefined" ? self : this);
