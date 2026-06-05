/**
 * Reverse-channel websocket endpoint (`/v1/reverse`).
 *
 * Solves issue #496. The default passthrough path denies every tool call,
 * feeds the model a placeholder "result", then resurfaces the real result a
 * turn later via session resume + content flattening — which drops the
 * tool_use↔tool_result linkage and desyncs the model (cached/wrong files,
 * lagged results, replay confusion) plus a 40–75x context spike.
 *
 * Instead, this endpoint keeps ONE socket open for an entire agent run and
 * drives ONE long-lived SDK `query()`. The query registers in-process MCP
 * tools whose handlers PARK until pi sends the real tool result back over the
 * same socket. The SDK loop therefore runs to completion in a single query:
 * no deny, no resume, no flatten, no cache spike.
 *
 * Division of labour (unchanged from pi's perspective):
 *   - pi still owns its agent loop, executes tools locally, renders UI, and
 *     persists sessions. It calls its streamFn once per turn.
 *   - meridian's single query owns the Claude/SDK session and relays each
 *     assistant turn to pi as Anthropic stream events.
 *
 * Tool-call correlation:
 *   The SDK's in-process MCP `tool()` handler receives the parsed input plus an
 *   `extra` arg whose `_meta["claudecode/toolUseId"]` carries the real Anthropic
 *   tool_use_id. The parking handler registers its resolver under that id; pi
 *   echoes the same id back as toolCallId, so deliver() resolves the exact call.
 *   No ordering assumption — correct for parallel and duplicate-name tools.
 */

import { randomUUID } from "node:crypto";
import type { Server } from "node:http";
import {
	createSdkMcpServer,
	query,
	tool,
} from "@anthropic-ai/claude-agent-sdk";
import { createNodeWebSocket } from "@hono/node-ws";
import type { Hono } from "hono";
import type { WSContext } from "hono/ws";
import { z } from "zod";
import { claudeLog } from "../logger";
import { resolveSdkWorkingDirectory } from "./cwd";
import { mapModelToClaudeModel, resolveClaudeExecutableAsync } from "./models";
import { buildQueryOptions } from "./query";
import { ensureFreshToken } from "./tokenRefresh";
import { BLOCKED_BUILTIN_TOOLS } from "./tools";
import {
	createRequestContext,
	runTransformHook,
	type Transform,
} from "./transform";

/** MCP server name the parking tools are registered under. */
const REVERSE_MCP_NAME = "reverse";
const REVERSE_MCP_PREFIX = `mcp__${REVERSE_MCP_NAME}__`;
const HEARTBEAT_MS = 15_000;

/** Key in the SDK MCP handler's `extra._meta` carrying the Anthropic
 *  tool_use_id. Present on @anthropic-ai/claude-agent-sdk's in-process MCP
 *  bridge; lets result correlation key by id rather than call order. */
const TOOL_USE_ID_META_KEY = "claudecode/toolUseId";

/** An MCP CallToolResult, as returned by an SDK in-process tool handler. */
interface CallToolResult {
	[key: string]: unknown;
	content: Array<{ type: "text"; text: string }>;
	isError?: boolean;
}

/** A pi tool schema, as carried in the `start` frame. */
interface PiTool {
	name: string;
	// pi's Tool type names the description field `desc`; tolerate `description` too.
	desc?: string;
	description?: string;
	// pi names the JSON-Schema field `parameters`; tolerate `inputSchema` too.
	parameters?: JsonSchema;
	inputSchema?: JsonSchema;
}

interface JsonSchema {
	type?: string;
	properties?: Record<string, JsonSchema & { description?: string }>;
	required?: string[];
	items?: JsonSchema;
	enum?: unknown[];
}

interface StartFrame {
	type: "start";
	systemPrompt?: string;
	messages: PiMessage[];
	tools?: PiTool[];
	model: string;
	options?: { maxTokens?: number; temperature?: number; reasoning?: unknown };
}

interface PiMessage {
	role: "user" | "assistant" | "toolResult" | "system";
	content?: unknown;
	toolCallId?: string;
	isError?: boolean;
}

interface ToolResultsFrame {
	type: "toolResults";
	results: Array<{ toolCallId: string; content: unknown; isError?: boolean }>;
}

/** Per-socket run state. One ReverseSession backs one held-open websocket. */
class ReverseSession {
	readonly resolvers = new Map<string, (result: CallToolResult) => void>();
	/** Results delivered by pi before their handler parked, keyed by tool_use_id. */
	readonly earlyResults = new Map<
		string,
		{ content: unknown; isError?: boolean }
	>();
	readonly abortController = new AbortController();
	/** Tool names exposed to the SDK, in `mcp__reverse__<bare>` form. */
	readonly toolNames: string[];
	private readonly mcpServer: ReturnType<typeof createSdkMcpServer>;

	constructor(tools: PiTool[]) {
		const sdkTools = tools.map((t) =>
			tool(
				t.name,
				t.desc ?? t.description ?? t.name,
				jsonSchemaToZodShape(t.parameters ?? t.inputSchema),
				(args, extra) => this.runTool(t.name, args, extra),
			),
		);
		this.mcpServer = createSdkMcpServer({
			name: REVERSE_MCP_NAME,
			version: "1.0.0",
			tools: sdkTools,
		});
		this.toolNames = tools.map((t) => `${REVERSE_MCP_PREFIX}${t.name}`);
	}

	get server(): ReturnType<typeof createSdkMcpServer> {
		return this.mcpServer;
	}

	/**
	 * Parking handler body. Correlate by the real Anthropic tool_use_id, exposed
	 * by the SDK at extra._meta["claudecode/toolUseId"]. pi echoes that id back as
	 * toolCallId, so deliver() resolves this exact call — no ordering assumption.
	 */
	private async runTool(
		bareName: string,
		_args: unknown,
		extra: unknown,
	): Promise<CallToolResult> {
		const id = extractToolUseId(extra);
		if (!id) {
			claudeLog("reverse.tool_missing_id", { tool: bareName });
			return {
				content: [
					{ type: "text", text: "reverse-channel: missing tool_use id" },
				],
				isError: true,
			};
		}
		// A result may already be buffered if pi answered before this handler parked.
		const early = this.earlyResults.get(id);
		if (early) {
			this.earlyResults.delete(id);
			return {
				content: toMcpContent(early.content),
				isError: early.isError === true,
			};
		}
		return new Promise<CallToolResult>((resolve) => {
			if (this.abortController.signal.aborted) {
				resolve({
					content: [{ type: "text", text: "aborted" }],
					isError: true,
				});
				return;
			}
			this.resolvers.set(id, resolve);
			this.abortController.signal.addEventListener(
				"abort",
				() => {
					if (this.resolvers.delete(id)) {
						resolve({
							content: [{ type: "text", text: "aborted" }],
							isError: true,
						});
					}
				},
				{ once: true },
			);
		});
	}

	/** Deliver a tool result from pi to the matching parked handler (by id). */
	deliver(result: {
		toolCallId: string;
		content: unknown;
		isError?: boolean;
	}): void {
		const resolve = this.resolvers.get(result.toolCallId);
		if (!resolve) {
			// Result arrived before the handler parked; buffer for runTool to pick up.
			this.earlyResults.set(result.toolCallId, {
				content: result.content,
				isError: result.isError,
			});
			return;
		}
		this.resolvers.delete(result.toolCallId);
		resolve({
			content: toMcpContent(result.content),
			isError: result.isError === true,
		});
	}

	abort(): void {
		if (!this.abortController.signal.aborted) this.abortController.abort();
	}
}

/** Convert a JSON-Schema object node into the SDK's ZodRawShape. */
function jsonSchemaToZodShape(
	schema: JsonSchema | undefined,
): Record<string, z.ZodTypeAny> {
	const shape: Record<string, z.ZodTypeAny> = {};
	const props = schema?.properties;
	if (!props) return shape;
	const required = new Set(schema?.required ?? []);
	for (const [key, prop] of Object.entries(props)) {
		let zodType = jsonSchemaNodeToZod(prop);
		if (prop.description) zodType = zodType.describe(prop.description);
		shape[key] = required.has(key) ? zodType : zodType.optional();
	}
	return shape;
}

function jsonSchemaNodeToZod(node: JsonSchema): z.ZodTypeAny {
	if (Array.isArray(node.enum) && node.enum.length > 0) return z.any();
	switch (node.type) {
		case "string":
			return z.string();
		case "number":
		case "integer":
			return z.number();
		case "boolean":
			return z.boolean();
		case "array":
			return z.array(node.items ? jsonSchemaNodeToZod(node.items) : z.any());
		case "object":
			return z.object(jsonSchemaToZodShape(node));
		default:
			return z.any();
	}
}

/** Pull the Anthropic tool_use_id the SDK exposes via the MCP handler's extra._meta. */
function extractToolUseId(extra: unknown): string | undefined {
	if (extra && typeof extra === "object") {
		const meta = (extra as { _meta?: Record<string, unknown> })._meta;
		const id = meta?.[TOOL_USE_ID_META_KEY];
		if (typeof id === "string") return id;
	}
	return undefined;
}

/** Normalize a pi tool-result payload into MCP content blocks. */
function toMcpContent(content: unknown): CallToolResult["content"] {
	if (typeof content === "string") return [{ type: "text", text: content }];
	if (Array.isArray(content)) {
		const blocks: CallToolResult["content"] = [];
		for (const block of content) {
			if (block && typeof block === "object") {
				const b = block as Record<string, unknown>;
				if (b.type === "text" && typeof b.text === "string") {
					blocks.push({ type: "text", text: b.text });
				} else {
					// pi tool results are text in practice; serialize any non-text block
					// (image/document/resource) rather than widen the SDK content union.
					blocks.push({ type: "text", text: JSON.stringify(b) });
				}
			} else {
				blocks.push({ type: "text", text: String(block) });
			}
		}
		return blocks.length > 0 ? blocks : [{ type: "text", text: "" }];
	}
	if (content == null) return [{ type: "text", text: "" }];
	return [{ type: "text", text: JSON.stringify(content) }];
}

/**
 * Seed the SDK query from pi's message history. A reverse run opens a fresh SDK
 * session, so prior assistant/tool turns are flattened into the prompt the same
 * way meridian flattens for any fresh (non-resume) session: assistant content
 * becomes `Assistant: <text>` and tool results become plain `Human:` text.
 */
function messagesToPrompt(messages: PiMessage[]): string {
	return messages
		.map((m) => {
			if (m.role === "assistant") {
				const text = flattenContentText(m.content);
				return text ? `Assistant: ${text}` : "";
			}
			if (m.role === "toolResult") {
				const text = flattenContentText(m.content);
				return text ? `Human: ${text}` : "";
			}
			if (m.role === "user") {
				const text = flattenContentText(m.content);
				return text ? `Human: ${text}` : "";
			}
			return "";
		})
		.filter(Boolean)
		.join("\n\n");
}

function flattenContentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((b) => {
			if (b && typeof b === "object") {
				const block = b as Record<string, unknown>;
				if (block.type === "text" && typeof block.text === "string")
					return block.text;
			}
			return "";
		})
		.filter(Boolean)
		.join("\n");
}

let cachedExecutable = "";

/**
 * Register the `/v1/reverse` websocket endpoint on the Hono app. Returns the
 * node-ws `injectWebSocket` function, which the server bootstrap must call on
 * the `http.Server` returned by `serve()`.
 */
export function registerReverseChannel(
	app: Hono,
	getPluginTransforms: () => readonly Transform[],
): (server: Server) => void {
	const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });

	app.get(
		"/v1/reverse",
		upgradeWebSocket(() => {
			let session: ReverseSession | null = null;
			let heartbeat: ReturnType<typeof setInterval> | null = null;

			const safeSend = (ws: WSContext, payload: unknown): void => {
				try {
					ws.send(JSON.stringify(payload));
				} catch (error) {
					claudeLog("reverse.send_failed", {
						error: error instanceof Error ? error.message : String(error),
					});
				}
			};

			return {
				onOpen(_event, ws) {
					heartbeat = setInterval(
						() => safeSend(ws, { ping: true }),
						HEARTBEAT_MS,
					);
				},

				async onMessage(event, ws) {
					let frame: StartFrame | ToolResultsFrame | { type: "abort" };
					try {
						frame = JSON.parse(String(event.data)) as typeof frame;
					} catch {
						return;
					}

					if (frame.type === "start") {
						if (session) return;
						session = new ReverseSession(frame.tools ?? []);
						void driveQuery(ws, session, frame, safeSend, getPluginTransforms);
						return;
					}
					if (frame.type === "toolResults" && session) {
						for (const result of frame.results) session.deliver(result);
						return;
					}
					if (frame.type === "abort" && session) {
						session.abort();
					}
				},

				onClose() {
					if (heartbeat) clearInterval(heartbeat);
					session?.abort();
				},

				onError() {
					if (heartbeat) clearInterval(heartbeat);
					session?.abort();
				},
			};
		}),
	);

	return injectWebSocket;
}

/**
 * Drive the single long-lived query for one socket: relay every assistant turn
 * as Anthropic stream events; the parking handlers stall the iterator between
 * turns until pi returns results. Strips the `mcp__reverse__` prefix from
 * tool_use names and queues each block's id for its handler as it relays.
 */
async function driveQuery(
	ws: WSContext,
	session: ReverseSession,
	start: StartFrame,
	safeSend: (ws: WSContext, payload: unknown) => void,
	getPluginTransforms: () => readonly Transform[],
): Promise<void> {
	try {
		await ensureFreshToken().catch(() => {
			/* reactive 401 path handles */
		});
		if (!cachedExecutable)
			cachedExecutable = await resolveClaudeExecutableAsync();

		const model = mapModelToClaudeModel(start.model);
		const cwd = resolveSdkWorkingDirectory({
			envOverride:
				process.env.MERIDIAN_WORKDIR ?? process.env.CLAUDE_PROXY_WORKDIR,
			adapterCwd: undefined,
			fallback: process.cwd(),
		}).workingDirectory;

		// Strip the proxy-loopback vars so the SDK subprocess uses Claude Max OAuth
		// instead of calling back through meridian (or a stale ANTHROPIC_BASE_URL).
		const cleanEnv = { ...process.env };
		delete cleanEnv.ANTHROPIC_API_KEY;
		delete cleanEnv.ANTHROPIC_BASE_URL;
		delete cleanEnv.ANTHROPIC_AUTH_TOKEN;

		// Run the user plugin pipeline (e.g. pi-scrub) over the system prompt with
		// adapter "pi", mirroring the normal HTTP path. The reverse path otherwise
		// bypasses transforms, leaking pi fingerprints into the prompt and tripping
		// Anthropic's third-party-app usage gate. Plugin transforms only — the pi
		// adapter's built-ins configure tool-blocking/passthrough the reverse
		// design overrides on purpose.
		const scrubbed = runTransformHook(
			getPluginTransforms(),
			"onRequest",
			createRequestContext({
				adapter: "pi",
				body: {},
				headers: new Headers(),
				model: start.model,
				messages: start.messages,
				systemContext: start.systemPrompt ?? "",
				tools: start.tools,
				stream: true,
				workingDirectory: cwd,
			}),
			"pi",
		);

		const built = buildQueryOptions({
			prompt: messagesToPrompt(scrubbed.messages as PiMessage[]),
			model,
			workingDirectory: cwd,
			systemContext: scrubbed.systemContext ?? "",
			claudeExecutable: cachedExecutable,
			passthrough: false,
			stream: true,
			sdkAgents: {},
			cleanEnv,
			hasDeferredTools: false,
			isUndo: false,
			blockedTools: [...BLOCKED_BUILTIN_TOOLS],
			incompatibleTools: [],
			mcpServerName: REVERSE_MCP_NAME,
			allowedMcpTools: session.toolNames,
		});

		// Replace the default internal MCP server with the parking server and
		// permit exactly its tools. permissionMode "bypassPermissions" (set by
		// buildQueryOptions) means the handlers fire with no gate.
		built.options.mcpServers = { [REVERSE_MCP_NAME]: session.server };
		built.options.allowedTools = session.toolNames;
		// Strip the SDK's built-in tool catalog (Read/Write/Bash/Grep/Edit/...) so
		// the model is offered ONLY pi's tools. Without this the scrubbed prompt
		// reads as Claude Code and the model reaches for native `Read` etc., which
		// pi can't execute — and which must never run against meridian's own
		// filesystem. Mirrors the passthrough branch in buildQueryOptions (#489);
		// blockedTools above also disallows them at runtime as defence in depth.
		(built.options as { tools?: unknown[] }).tools = [];
		built.options.includePartialMessages = true;
		delete (built.options as { hooks?: unknown }).hooks;
		(built.options as { abortController?: AbortController }).abortController =
			session.abortController;

		claudeLog("reverse.start", { model, tools: session.toolNames.length });
		console.error(
			`[PROXY] reverse-channel start model=${model} tools=${session.toolNames.length}`,
		);
		for await (const message of query(built)) {
			if (session.abortController.signal.aborted) break;
			if ((message as { type?: string }).type !== "stream_event") continue;
			const event = (message as { event: AnthropicStreamEvent }).event;
			relayEvent(event);
			safeSend(ws, event);
		}
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		// An abort (socket close, pi cancel, or `abort` frame) makes the SDK query
		// throw "Claude Code process aborted by user". That is an expected end of a
		// reverse run, not a failure — log it quietly and skip the client error.
		if (session.abortController.signal.aborted) {
			claudeLog("reverse.aborted", { error: message });
			return;
		}
		const detail =
			error instanceof Error ? (error.stack ?? error.message) : String(error);
		claudeLog("reverse.query_failed", { error: message });
		console.error(`[PROXY] reverse-channel query failed: ${detail}`);
		safeSend(ws, {
			type: "error",
			msg: message || "reverse-channel query failed (see meridian logs)",
		});
	} finally {
		session.abort();
		try {
			ws.close(1000, "done");
		} catch {
			/* already closed */
		}
	}
}

interface AnthropicStreamEvent {
	type?: string;
	index?: number;
	content_block?: { type?: string; id?: string; name?: string };
}

/**
 * Pre-relay mutation: for tool_use blocks, strip the `mcp__reverse__` prefix so
 * pi sees its own bare tool name. Result correlation is by tool_use_id in the
 * parking handler, so no id bookkeeping happens here.
 */
function relayEvent(event: AnthropicStreamEvent): void {
	if (event.type !== "content_block_start") return;
	const block = event.content_block;
	if (!block || block.type !== "tool_use" || typeof block.name !== "string")
		return;
	block.name = block.name.startsWith(REVERSE_MCP_PREFIX)
		? block.name.slice(REVERSE_MCP_PREFIX.length)
		: block.name;
}
