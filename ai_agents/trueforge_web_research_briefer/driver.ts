/**
 * Web research briefer - TrueForge driver.
 *
 * Opens a session on the saved `web-research-brief` agent, streams one research
 * turn, and prints a labeled trace of every step the harness takes, then the
 * final brief and per-turn metrics. That trace is the source material for the
 * article's Loop and Trace sections.
 *
 * Run (with the TrueForge server running and the agent saved):
 *   npm start -- "your research question"
 *
 * Verified against https://trueforge.dev/api/use-agent
 */

import "dotenv/config";
import {
  TrueForge,
  TrueForgeApi,
  isEventDelta,
  mergeEventDelta,
} from "@truefoundry/trueforge-sdk";

const BASE_URL = process.env.TRUEFORGE_BASE_URL ?? "http://localhost:8790";
const AGENT_NAME = process.env.TRUEFORGE_AGENT ?? "web-research-brief";
const TOKEN = process.env.TRUEFORGE_TOKEN; // only needed for hosted mode (OIDC)

const DEFAULT_QUESTION =
  "Compare the open-source LLM observability tools Langfuse, Arize Phoenix, and Helicone on features, self-hosting, and pricing, then write a one-page brief with sources.";

const question = process.argv.slice(2).join(" ").trim() || DEFAULT_QUESTION;

const client = new TrueForge({
  baseUrl: BASE_URL,
  timeoutInSeconds: 600, // research turns run long; do not lower this
  ...(TOKEN ? { token: TOKEN } : {}),
});

// Per-stream event index, keyed by event id. Model output arrives as an empty
// `model.message` base followed by `model.message.delta` fragments that share
// the base id; we merge deltas into the base so `events` always holds the full
// message. The pause handlers below look tool calls up in this same index.
const events = new Map<string, TrueForgeApi.TurnStreamingEvent>();

let turnId: string | undefined;
let lastSequenceNumber = 0;

function log(label: string, detail = ""): void {
  console.log(`\n[${label}]${detail ? " " + detail : ""}`);
}

async function main(): Promise<void> {
  console.log(`TrueForge   : ${BASE_URL}`);
  console.log(`Agent       : ${AGENT_NAME}`);
  console.log(`Question    : ${question}\n`);

  const { data: session } = await client.sessions.create({
    agent: { name: AGENT_NAME },
  });
  console.log(`Session     : ${session.id}`);

  const stream = await client.sessions.createTurnStream(session.id, {
    input: [{ type: "user.message", content: question }],
  });

  for await (const { data: event, id } of stream.withMetadata()) {
    if (id != null) lastSequenceNumber = Number(id);

    // Keep the event index current (merge deltas, store everything else).
    if (isEventDelta(event)) {
      const base = events.get(event.id);
      if (base) mergeEventDelta(base, event);
    } else {
      events.set(event.id, event);
    }

    switch (event.type) {
      case "turn.created":
        turnId = event.turnId;
        log("turn.created", `turn ${event.turnId}`);
        break;

      case "mcp.initialize":
        log(
          "mcp.initialize",
          (event.mcpServers ?? []).map((s: { name?: string }) => s.name).join(", "),
        );
        break;

      case "sandbox.created":
        log("sandbox.created", event.sandboxId ?? "");
        break;

      case "thread.created":
        log("subagent", `${event.title} (thread ${event.threadId})`);
        break;

      case "thread.done":
        log("subagent.done", `${event.title} (thread ${event.threadId})`);
        break;

      case "tool.response":
        log("tool.response", `call ${event.toolCallId}`);
        break;

      case "model.message.delta":
        // Stream only the root agent's reply to stdout; subagents run in their
        // own threads and their intermediate output stays out of the console.
        if (event.threadId === "main" && event.content) {
          process.stdout.write(event.content);
        }
        break;

      case "tool.approval_required":
        // web-research-brief has no gated tool, so this never fires here. When
        // you attach a write connector and gate it, resume with a new turn
        // carrying a `user.tool_approval` per pending call (see README).
        log("tool.approval_required", "would pause for human sign-off");
        break;

      case "turn.done": {
        const state = event.state;
        log("turn.done", `status ${state.status}`);
        if (state.status === "done") {
          console.log("\n----- brief -----\n");
          console.log(state.output?.content ?? "(no text output)");
          const m = state.metrics;
          if (m) {
            console.log("\n----- metrics -----");
            console.log(`tokens : ${m.totalTokens ?? "?"}`);
            console.log(`cost   : $${m.totalCostInUsd ?? "?"}`);
          }
        } else if (state.status === "error") {
          console.error(`\nTurn error: ${state.message}`);
        }
        break;
      }

      default:
        break;
    }
  }

  console.log(`\nLast sequence number: ${lastSequenceNumber}`);
  console.log(`Turn id             : ${turnId ?? "(none)"}`);
}

main().catch((err) => {
  console.error("\nDriver failed:", err);
  process.exit(1);
});
