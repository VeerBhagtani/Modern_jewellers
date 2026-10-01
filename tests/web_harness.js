// Node harness for tests/test_web.py: runs web/arouse.js on requests from a JSON file and prints JSON.
"use strict";
const fs = require("fs");
const path = require("path");
const { Arouse } = require(path.join(__dirname, "..", "web", "arouse.js"));

const req = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const dir = req.model_dir;
const cfg = JSON.parse(fs.readFileSync(path.join(dir, "config.json"), "utf8"));
const tok = new Arouse.Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, "tokenizer.json"), "utf8")));
const out = {};
if (fs.existsSync(path.join(dir, "knowledge.json"))) Arouse.setKnowledge(JSON.parse(fs.readFileSync(path.join(dir, "knowledge.json"), "utf8")));
const at0 = (s) => { const [d, t] = s.split("T"); const [y, mo, dd] = d.split("-").map(Number); const [h, mi] = t.split(":").map(Number); return new Date(y, mo - 1, dd, h, mi); };
if (req.replay) { // run every gold tool call of each episode through the JS sandbox
  out.replay = req.replay.map((ep) => {
    const sb = new Arouse.Sandbox({ reminders: ep.reminders, files: ep.files, failures: ep.failures });
    sb.now = () => at0(ep.now);
    const results = [];
    for (const ev of ep.events) {
      if (ev.type === "arouse" && ev.turn.action.type === "tool_call") {
        const [ok, payload] = sb.execute(ev.turn.action.tool, ev.turn.action.arguments);
        results.push({ type: ok ? "tool_result" : "tool_error", content: payload });
      }
    }
    return results;
  });
}
if (req.kb) out.kb = req.kb.map((q) => Arouse.kbSearch(q));

if (req.tokenize) out.tokenize = req.tokenize.map((t) => tok.encode(t));
if (req.answers) out.answers = req.answers.map((evs) => Arouse.answerFromResult(evs));
const vocabAll = new Set(fs.readFileSync(path.join(dir, "response_vocab.txt"), "utf8").split(/\s+/).filter(Boolean));
if (req.answer_issue) out.answer_issue = req.answer_issue.map(([action, evs]) => Arouse.answerIssue(action, evs, vocabAll) !== null);
if (req.grounding) out.grounding = req.grounding.map(([action, evs]) => Arouse.groundingIssue(action, evs) !== null);
if (req.complete) {
  out.complete = req.complete.map(([evs, ctx, field, value]) => new Arouse.CopyConstraint(tok, evs, ctx).complete(field, value));
}
if (req.continuations) {
  out.continuations = req.continuations.map(([evs, ctx, field, partial]) =>
    new Arouse.CopyConstraint(tok, evs, ctx).continuations(field, partial).sort());
}

if (req.logits || req.decisions || req.episodes) {
  const raw = fs.readFileSync(path.join(dir, "weights.bin"));
  const model = new Arouse.Model(cfg, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length));
  const engine = new Arouse.Engine(model, tok);
  const vocab = new Set(fs.readFileSync(path.join(dir, "response_vocab.txt"), "utf8").split(/\s+/).filter(Boolean));
  const at = (s) => { const [d, t] = s.split("T"); const [y, mo, dd] = d.split("-").map(Number); const [h, mi] = t.split(":").map(Number); return new Date(y, mo - 1, dd, h, mi); };
  const header = (ep) => ({ context: Arouse.buildContext(at(ep.now), req.timezone || "Asia/Kolkata"), tools: Arouse.TOOL_NAMES });

  if (req.logits) {
    out.logits = req.logits.map((ids) => {
      engine.cached = []; model.len = 0;
      let lg;
      for (const t of ids) lg = model.step(t);
      return Array.from(lg);
    });
  }
  if (req.prompts) out.prompts = req.prompts.map(([ep, n]) => Arouse.encodePrompt(tok, header(ep), ep.events.slice(0, n)));
  if (req.decisions) {
    const rt = new Arouse.Runtime(engine, { vocab });
    out.decisions = req.decisions.map(([ep, n, mode]) => {
      Object.assign(rt, mode === "raw" ? { retries: 0, guardCompletion: false, constrainCopy: false }
        : { retries: 2, guardCompletion: true, constrainCopy: true });
      const r = rt.nextTurn(header(ep), ep.events.slice(0, n));
      return { action: r.turn.action, attempts: r.attempts, valid: r.valid };
    });
  }
  if (req.episodes) {
    const rt = new Arouse.Runtime(engine, { vocab });
    out.episodes = req.episodes.map(([ep, n]) => {
      const sb = new Arouse.Sandbox({ reminders: ep.reminders, files: ep.files, failures: ep.failures });
      sb.now = () => at(ep.now);
      const res = rt.run(header(ep), ep.events.slice(0, n), (tool, args) => sb.execute(tool, args));
      return { final: res.final, attempts: res.turns.map((t) => t.attempts), state: sb.snapshot() };
    });
  }
}
process.stdout.write(JSON.stringify(out));
