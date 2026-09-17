import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function registerAihbProvider(pi: ExtensionAPI): void {
	pi.registerProvider("aibh", {
		baseUrl: "https://aibh.cc/v1",
		apiKey: "$OPENAI_API_KEY",
		api: "openai-completions",
		models: [
			{
				id: "v4 pro",
				name: "AIHB v4 pro",
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 128000,
				maxTokens: 4096,
			},
		],
	});
}
