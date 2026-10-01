/* Runs Arouse off the main thread. Messages:
 *   in:  {type: "load", base}                         -> progress*, ready | error
 *   in:  {type: "run", id, events, state, timezone}   -> step*, done | error
 */
importScripts("arouse.js");

let arouse = null;

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      arouse = await Arouse.loadArouse(msg.base || "model/", (p) => self.postMessage({ type: "progress", p }));
      const c = arouse.cfg;
      self.postMessage({ type: "ready", info: { name: c.name, params: c.num_parameters, steps: c.train_steps,
        context: c.context_length, notes: c.notes } });
    } else if (msg.type === "run") {
      if (!arouse) throw new Error("model not loaded");
      const sb = new Arouse.Sandbox(msg.state || {});
      const header = { context: Arouse.buildContext(sb.now(), msg.timezone || "local"), tools: Arouse.TOOL_NAMES };
      const t0 = performance.now();
      const res = arouse.runtime.run(header, msg.events, (tool, args) => sb.execute(tool, args),
        (event) => self.postMessage({ type: "step", id: msg.id, event }));
      self.postMessage({ type: "done", id: msg.id, events: res.events, final: res.final, state: sb.state(),
        ms: Math.round(performance.now() - t0) });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: String(err && err.message || err) });
  }
};
